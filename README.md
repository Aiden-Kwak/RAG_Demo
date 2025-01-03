# RAG_Demo

```bash
.env: DEEPSEEK_API_KEY, OPENAI_API_KEY
Usage: python main.py <Claim ID>
```

<h3>20250103</h3>
임베딩이 완료될 기미가 안보임. 아래에선 45,000개 행 데이터프레임 임베딩에 6시간.<br>
https://community.openai.com/t/semantic-embedding-super-slow-text-embedding-ada-002/42183<br>
- latelimit 기준 검토<br>
- 병렬처리 시도<br>
