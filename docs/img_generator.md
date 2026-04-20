```bash 
!python khmer_ocr_generator.py \
  --input test/ \
  --fonts \
    fonts/KantumruyPro-VariableFont_wght.ttf \
    fonts/Khmer-Regular.ttf \
    fonts/khmerOS.ttf \
    fonts/Koulen-Regular.ttf \
    fonts/Moul-Regular.ttf \
    fonts/Nokora-VariableFont_wght.ttf \
    "fonts/NotoSansKhmer-VariableFont_wdth,wght.ttf" \
    fonts/Siemreap-Regular.ttf \
  --output ./ocr_data \
  --samples 5000 \
  --font_sizes 24 28 32 36 \
  --augment \
  --seed 2024
```

/home/thareah/Desktop/TrOCR/thaocr/tools/generateimg/img_generator.py