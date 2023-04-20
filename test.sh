python src/main.py --add-geom --add-embs --add-eweights \
  --src-data FUNSD  \
  --edge-type fully --node-granularity gt \
  --num-polar-bins 8 \
  --gpu 0 \
  --test \
  --weights e2e-20230412-0529.pt
