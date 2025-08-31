import argparse
import re
import torch
import requests
from io import BytesIO

from PIL import Image
from transformers import CLIPImageProcessor, TextStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


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
    
    # 토큰라이저 패딩 토큰 설정
    if getattr(tokenizer, 'pad_token_id', None) is None:
        if getattr(tokenizer, 'eos_token_id', None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback: 일반적인 EOS 토큰 ID 사용
            tokenizer.pad_token_id = 2  # 대부분 모델에서 </s> 토큰
    
    # 비전 타워 강제 초기화 (AttributeError 방지)
    try:
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            if hasattr(vision_tower, 'is_loaded') and not vision_tower.is_loaded:
                vision_tower.load_model(device_map=args.device)
            # 비전 타워 내부 모듈 확인 및 초기화
            if hasattr(vision_tower, 'vision_tower') and vision_tower.vision_tower is None:
                vision_tower.load_model(device_map=args.device)
    except Exception as e:
        print(f"Warning: Vision tower initialization issue: {e}")

    # 비전 타워 및 이미지 프로세서 설정
    if image_processor is None:
        try:
            # 비전 타워 로드 및 초기화
            vision_tower = model.get_vision_tower()
            if hasattr(vision_tower, 'is_loaded') and not vision_tower.is_loaded:
                vision_tower.load_model(device_map=args.device)
                
            # 이미지 프로세서 가져오기
            image_processor = getattr(vision_tower, 'image_processor', None)
            
            # Fallback to config-based processor
            if image_processor is None:
                mm_vision_tower = getattr(model.config, 'mm_vision_tower', None)
                if mm_vision_tower:
                    image_processor = CLIPImageProcessor.from_pretrained(mm_vision_tower)
                    
            # Final fallback
            if image_processor is None:
                image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
                
        except Exception as e:
            print(f"Warning: Could not load image processor: {e}")
            try:
                image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
            except Exception:
                print("Failed to load fallback image processor")
                return

    def clean_output(text: str) -> str:
        """출력 텍스트를 깔끔하게 정리"""
        # 불필요한 마크업 제거
        cleanup_patterns = [
            "### assistant", "### Assistant", "### assistance", 
            "### assidance", "### assisistant", "###"
        ]
        for pattern in cleanup_patterns:
            text = text.replace(pattern, "")
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        # 콜론과 공백으로 시작하는 경우 제거
        if text.startswith(": "):
            text = text[2:].strip()
        
        # 중복된 번호 매기기 제거
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        filtered = []
        seen_cores = set()
        for ln in lines:
            m = re.match(r"^(\d+)[\.)]\s*(.*)$", ln)
            core = m.group(2).strip() if m else ln
            if core not in seen_cores:
                seen_cores.add(core)
                filtered.append(core)
        
        result = "\n".join(filtered).strip()
        
        # 첫 번째 완전한 문단만 유지 (긴 반복 방지)
        paragraphs = result.split('\n\n')
        if len(paragraphs) > 1:
            result = paragraphs[0].strip()
            
        return result

    # Prefer midm conversation template when model/type indicates MIDM
    if ('midm' in model_name.lower()) or (getattr(model.config, 'model_type', '') == 'llava_midm'):
        conv_mode = "midm"
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

        # 매 대화마다 새로운 conversation 객체 생성 (이전 대화 누적하지 않음)
        conv = conv_templates[args.conv_mode].copy()
        
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
        else:
            # 이미지 없이 텍스트만 처리
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        
        # 텍스트 생성 설정
        stopping_criteria = KeywordsStoppingCriteria(["###"], tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 생성 파라미터
        gen_kwargs = {
            'inputs': input_ids,
            'attention_mask': attention_mask,
            'images': image_tensor,
            'do_sample': False,
            'max_new_tokens': min(args.max_new_tokens, 200),
            'streamer': streamer,
            'use_cache': True,
            'repetition_penalty': 1.3,
            'early_stopping': True,
            'stopping_criteria': [stopping_criteria],
        }
        
        # 토큰 ID 설정
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            gen_kwargs['pad_token_id'] = tokenizer.pad_token_id
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            gen_kwargs['eos_token_id'] = tokenizer.eos_token_id

        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        outputs = clean_output(outputs)

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
