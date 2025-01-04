# RAG 기반 사실 확인 시스템, RAG 연습용으로 제작중

## 개요

이 프로젝트는 **RAG(Retrieval-Augmented Generation)** 방식을 활용한 **사실 확인 시스템**이다. 신뢰할 수 있는 증거를 바탕으로 전문 수준의 사실 확인 결과를 생성하는걸 목표한다.

### 주요 기능:
- **주장 평가**: 문서에서 검색된 증거를 바탕으로 주장의 진위를 평가.
- **RAG 프레임워크**: 검색 시스템과 생성 모델을 통합해 사실 기반 결과를 제공.
- **유연한 파이프라인**: 다양한 주장과 문서 소스를 처리할 수 있도록 설계함.

---

## 작동 원리

### 1. 데이터 준비
PDF 및 관련 메타데이터를 처리하여 OpenAI 임베딩 모델을 사용해 검색 가능한 임베딩을 생성한다. 주요 단계는 다음과 같다:
- `PyPDF2`를 사용해 PDF의 텍스트와 페이지 번호를 추출.
- 추출한 텍스트를 임베딩 생성에 최적화된 조각(Chunk)으로 분할.
- OpenAI `text-embedding-3-small`를 이용해 임베딩을 생성.
- FAISS 인덱스에 임베딩을 저장하여 효율적인 검색이 가능하도록 함.

### 2. RAG(Retrieval-Augmented Generation) 방식
1. **검색기(Retriever)**:
   - FAISS를 이용해 주어진 주장에 가장 관련성 높은 증거를 검색한다.
   - 증거와 관련된 메타데이터를 통해 결과의 맥락을 제공한다.
2. **생성기(Generator)**:
   - 검색된 증거를 사용해 구조화된 프롬프트를 생성.
   - DeepSeek Chat API을 사용해 증거 기반 결과를 생성.

### 3. 사실 확인 파이프라인
전체 프로세스는 다음과 같다:
1. **주장 입력**: 주장과 추가적인 맥락 정보를 입력.
2. **증거 검색**: FAISS를 통해 가장 관련성 높은 증거를 검색.
3. **프롬프트 구성**: 검색된 증거를 기반으로 구조화된 프롬프트를 생성.
4. **주장 평가**: 생성 모델을 통해 사실 확인 보고서를 생성하며, 이 보고서는 증거를 인용한다.

---

## 프로젝트 구조
```bash
.
├── main.py             # 메인 스크립트
├── info.json           # 문서 메타데이터 파일 (사전에 수기작성함)
├── claim.json          # 평가할 주장 데이터 파일 (climate feedback 출처기반으로 수기작성함.)
├── pdf/                # PDF 문서 저장디렉토리
│   └── 1.pdf           # 샘플 PDF 문서(IPCC 특별보고서)
├── faiss_index.bin     # FAISS 인덱스 파일 (생성됨)
├── metadata.json       # FAISS 인덱스용 메타데이터 파일 (생성됨)
├── requirements.txt    # 필요패키지
└── .env                # 환경 변수 파일 (API 키 및 설정)
```


### 환경 변수
`.env` 파일을 생성하고 다음 키를 추가:
```env
OPENAI_API_KEY=<your_openai_api_key>
DEEPSEEK_API_KEY=<your_deepseek_api_key>
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## 예시 출력 (claim_id: 1)

Claim: Earth not warming as predicted by (junk) climate models because plant photosynthesis is absorbing more CO2 than imagined. Climate change is a hoax.

Fact-check Result: ### Claim Evaluation:
This claim is **inaccurate** and **misleading**.

### Reasoning:
The claim that "Earth not warming as predicted by (junk) climate models because plant photosynthesis is absorbing more CO2 than imagined" is not supported by scientific evidence. While it is true that increased CO2 levels can enhance plant photosynthesis (a phenomenon known as CO2 fertilization), this effect is limited and does not negate the overall warming trend predicted by climate models. 

1. **Limitations of CO2 Fertilization**: The IPCC Special Report on Global Warming of 1.5°C (2018) highlights that while elevated CO2 can increase plant growth, this effect is constrained by factors such as nutrient availability (e.g., nitrogen and phosphorus), water stress, and temperature changes. These limitations mean that the terrestrial carbon sink cannot indefinitely offset rising CO2 emissions (IPCC, 2018, p. 234).

2. **Net Carbon Sink Reduction**: The same report notes that climate change is projected to reduce the carbon sink expected under CO2 increase alone. This is due to factors like increased decomposition rates, wildfires, and land-use changes, which can release stored carbon back into the atmosphere (IPCC, 2018, p. 234).

3. **Climate Models Are Reliable**: The claim that climate models are "junk" is unfounded. Climate models have been extensively validated against historical data and have consistently demonstrated their ability to predict global temperature trends. The IPCC report emphasizes that climate models are robust tools for understanding and projecting climate change (IPCC, 2018, p. 187).

4. **Climate Change Is Not a Hoax**: The overwhelming consensus among climate scientists, as reflected in the IPCC reports and numerous peer-reviewed studies, is that climate change is real, primarily driven by human activities, and poses significant risks to ecosystems and human societies.

### Evidence:
- **IPCC Special Report on Global Warming of 1.5°C (2018), p. 234**: "The projected net effect of climate change is to reduce the carbon sink expected under CO2 increase alone."
- **IPCC Special Report on Global Warming of 1.5°C (2018), p. 187**: "Climate models are robust tools for understanding and projecting climate change."
- **IPCC Special Report on Global Warming of 1.5°C (2018), p. 234**: "Nitrogen, phosphorus and other nutrients will limit the terrestrial carbon cycle response to both elevated CO2 and altered climate."

### Conclusion:
The claim that Earth is not warming as predicted because plants are absorbing more CO2 than imagined is inaccurate and misleading. While increased CO2 can enhance plant growth, this effect is limited by various factors and does not counteract the overall warming trend. Climate models are reliable and have consistently predicted global warming, which is supported by extensive scientific evidence. Climate change is a well-documented phenomenon, and the assertion that it is a hoax is not supported by the scientific community.
