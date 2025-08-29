## 📊 Midm-2.0-Mini-Vision-Instruct

- **Midm-2.0-Mini-Vision-Instruct**은 Midm-2.0-Mini-Vision-Instruct은 한국어 이미지 인식에 특화된 고성능, 경량 Vision-Language Model입니다. K-intelligence/Midm-2.0-Mini-Instruct 기반으로 구축되어 한국어 텍스트가 포함된 이미지 이해와 한국어 응답 생성에 최적화되었습니다.
- **End-to-End** LLaVA 구조를 채택하여 이미지 입력부터 텍스트 출력까지 하나의 파이프라인에서 처리하며, 추가적인 중간 모델 없이 자연스럽게 멀티모달 처리를 지원합니다.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/653494138bde2fae198fe89e/NAGzLbylQfYIJN-JI4NBN.png)

### 📂 모델 접근
- **GitHub**: [bigdefence/midm-vision](https://github.com/bigdefence/midm-vision) 🌐
- **HuggingFace**: [bigdefence/Midm-2.0-Mini-Vision-Instruct](https://huggingface.co/bigdefence/Midm-2.0-Mini-Vision-Instruct) 🤗
- **모델 크기**: 2B 파라미터 📊

## 🌟 주요 특징

- **🇰🇷 한국어 특화**: 한국어 언어적 특성에 최적화
- **⚡ 경량화**: 2B 파라미터로 효율적인 추론 성능
- **🎯 고정확도**: 다양한 한국어 환경에서 우수한 성능
- **🔧 실용성**: 실시간 이미지 인식 애플리케이션에 적합

## 📋 모델 정보

| 항목 | 세부사항 |
|------|----------|
| **기반 모델** | K-intelligence/Midm-2.0-Mini-Instruct |
| **언어** | 한국어 (Korean) |
| **모델 크기** | ~2B 파라미터 |
| **작업 유형** | Image-to-Text 이미지 멀티모달 |
| **라이선스** | Apache 2.0 |

### 🔧 레포지토리 다운로드 및 환경 설정

**Midm-2.0-Mini-Vision-Instruct**을 시작하려면 다음과 같이 레포지토리를 클론하고 환경을 설정하세요. 🛠️

1. **레포지토리 클론**:
   ```bash
   git clone https://github.com/bigdefence/midm-vision
   cd midm-vision
   ```

2. **의존성 설치**:
   ```bash
   conda create -n midm-vision python=3.10 -y
   conda activate midm-vision
   pip install -e .
   ```

### 📥 다운로드 방법

**Huggingface CLI 사용**:
```bash
pip install -U huggingface_hub
huggingface-cli download bigdefence/Midm-Vision --local-dir ./checkpoints
```

**Snapshot Download 사용**:
```bash
pip install -U huggingface_hub
```
```python
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="bigdefence/Midm-Vision",
  local_dir="./checkpoints",
  resume_download=True
)
```

**Git 사용**:
```bash
git lfs install
git clone https://huggingface.co/bigdefence/midm-vision
```

### 🔄 로컬 추론

**Midm-Vision**으로 추론을 수행하려면 다음 단계를 따라 모델을 설정하고 로컬에서 실행하세요. 📡

1. **모델 준비**:
   - [HuggingFace](https://huggingface.co/bigdefence/Midm-2.0-Mini-Vision-Instruct)에서 **Midm-2.0-Mini-Vision-Instruct** 다운로드 📦

2. **추론 실행**:
     - **Streaming**
     ```bash
     python3 infer.py --model-path checkpoints --image-file test.jpg
     ```

## 🔧 훈련 세부사항

### 훈련 설정
- **Base Model**: K-intelligence/Midm-2.0-Mini-Instruct
- **Hardware**: 4x NVIDIA RTX 4090 GPU
- **Training Time**: 10시간

## 📜 라이선스

이 모델은 Apache 2.0 라이선스 하에 배포됩니다. 상업적 사용이 가능하며, 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.


## 📞 문의사항

- **개발**: BigDefence
