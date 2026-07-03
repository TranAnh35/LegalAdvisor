# LegalAdvisor Mini Scripts

Thu muc nay chi giu cac script can thiet de tai du lieu, tao chunks, build artifact phu tro va chay ban RAG mini. Pipeline training, evaluation, benchmark va crawler da duoc luoc bo khoi branch nay.

## dataset/

- `download.py`: tai dataset Zalo Legal ve `data/raw/zalo_ai_legal_text_retrieval/`.
- `normalize.py`: tien ich normalize Unicode cho corpus khi can debug rieng. Runtime khong doc truc tiep file nay.

## Preprocess

Tao file chunks dung cho runtime:

```bash
python scripts/zalo_legal_preprocess.py
```

Output mac dinh:

```text
data/processed/zalo-legal/chunks_schema.jsonl
```

## utils/

- `extract_citations.py`: tien ich kiem tra/trich xuat trich dan luat.
- `export_act_codes.py`: xuat danh sach ma van ban trong du lieu.
- `build_law_registry.py`: tao `data/registry/law_registry.json` tu `chunks_schema.jsonl` de UI/citation hien thi ten van ban.

## Build index

Viec build FAISS index nam o module runtime:

```bash
python src/retrieval/build_index.py ^
  --chunks data/processed/zalo-legal/chunks_schema.jsonl ^
  --base-model intfloat/multilingual-e5-small ^
  --model-dir models/retrieval/base_encoder ^
  --output-dir models/retrieval/index
```

Sau khi build lai chunks, phai build lai index. Khong dung chung index cu voi chunks moi.

Neu muon build bang encoder local trong `models/retrieval/base_encoder`, them flag `--use-local-model`.