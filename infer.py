import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import re
from transformers import CLIPImageProcessor

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    # Ensure pad token is defined to avoid generation warnings
    if getattr(tokenizer, 'pad_token_id', None) is None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    # Fallback: ensure image_processor is available for MIDM/LLaVA
    if image_processor is None:
        try:
            vision_tower = model.get_vision_tower()
            if hasattr(vision_tower, 'is_loaded') and not vision_tower.is_loaded:
                vision_tower.load_model(device_map=args.device)
            image_processor = getattr(vision_tower, 'image_processor', None)
        except Exception:
            image_processor = None
    if image_processor is None:
        mm_vision_tower = getattr(model.config, 'mm_vision_tower', None)
        if mm_vision_tower:
            try:
                image_processor = CLIPImageProcessor.from_pretrained(mm_vision_tower)
            except Exception:
                image_processor = None
    if image_processor is None:
        try:
            image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        except Exception:
            pass

    def _dedupe_numbered_lines(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        filtered = []
        seen_cores = set()
        for ln in lines:
            m = re.match(r"^(\d+)[\.)]\s*(.*)$", ln)
            core = m.group(2).strip() if m else ln
            if core in seen_cores:
                continue
            seen_cores.add(core)
            filtered.append(core)
        return "\n".join(filtered).strip()

    # Prefer midm conversation template when model/type indicates MIDM
    if ('midm' in model_name.lower()) or (getattr(model.config, 'model_type', '') == 'llava_midm'):
        conv_mode = "midm"
    elif 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "synatra" in model_name.lower():
        conv_mode = "mistral"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    # Display-friendly roles for CLI prompt
    if args.conv_mode == "midm" or "mpt" in model_name.lower():
        display_roles = ('user', 'assistant')
    elif "synatra" in model_name.lower():
        display_roles = ('user', 'assistant')
    else:
        display_roles = conv.roles

    image = load_image(args.image_file)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{display_roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{display_roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        
        # ### 키워드로 생성 중단 설정
        stop_str = "###"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Use EOS tokens for MIDM/MPT-style templates to terminate generation
        # 강력한 반복 방지와 조기 종료 설정
        gen_kwargs = dict(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            do_sample=False,  # 그리디 생성으로 일관성 향상
            max_new_tokens=min(args.max_new_tokens, 200),  # 토큰 제한으로 긴 반복 방지
            streamer=streamer,
            use_cache=True,
            pad_token_id=(tokenizer.pad_token_id if getattr(tokenizer, 'pad_token_id', None) is not None else tokenizer.eos_token_id),
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=8,
            repetition_penalty=1.3,
            early_stopping=True,
            stopping_criteria=[stopping_criteria],
        )
        
        # MIDM 전용 EOS 토큰 설정
        eos_ids = []
        if tokenizer.eos_token_id is not None:
            eos_ids.append(int(tokenizer.eos_token_id))
        try:
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_id, int) and eot_id != -1 and eot_id not in eos_ids:
                eos_ids.append(eot_id)
        except Exception:
            pass
        try:
            end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
            if isinstance(end_header_id, int) and end_header_id != -1 and end_header_id not in eos_ids:
                eos_ids.append(end_header_id)
        except Exception:
            pass
        if len(eos_ids) > 0:
            gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]

        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        # 마크업과 반복 정리
        outputs = outputs.replace("### assistant", "").replace("### Assistant", "").replace("### assistance", "").replace("### assidance", "").replace("### assisistant", "").strip()
        if outputs.startswith(": "):
            outputs = outputs[2:].strip()
        outputs = _dedupe_numbered_lines(outputs)
        # 첫 번째 완전한 문단만 유지 (긴 반복 방지)
        sentences = outputs.split('\n\n')
        if len(sentences) > 1:
            outputs = sentences[0].strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
