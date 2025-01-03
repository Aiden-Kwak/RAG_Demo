# RAG_Demo

```bash
.env: DEEPSEEK_API_KEY, OPENAI_API_KEY
Usage: python main.py <Claim ID>
```

<h3>20250103</h3>
- 임베딩이 완료될 기미가 안보임. 아래에선 45,000개 행 데이터프레임 임베딩에 6시간.<br>
https://community.openai.com/t/semantic-embedding-super-slow-text-embedding-ada-002/42183<br>
- latelimit 기준 검토/ 완<br>
- 병렬처리 시도/ 완<br>
- 결과가 마음에 들지 않음. 반박하는 논리와 제시한 자료의 관련성이 적음. (result.md)스톡홀름과 런던의 혼잡 통행료 정책에 대한 자료를 가져왔음. <br>
- 구조 명확성 부족함.<br>
- 자료수집 에이전트를 따로 제작해서 체이닝해보자.<br>
- 메모: TOP_K 8개 우선 출력해보고 시작하자.

