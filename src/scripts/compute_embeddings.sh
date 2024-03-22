# fe_type="timm:vit_huge_patch14_clip_224.laion2b:1280"
fe_type="timm:vit_base_patch16_clip_224.openai:768"
gpu="5"

python src/scripts/compute_embeddings.py \
     --detailed_name \
     --fe_type $fe_type \
     --batch_size 1024 \
     --gpu $gpu \
     --model ICL \
     --image_embedding_cache_dir ../image_datasets/latest_imagenet/cached_embeddings/ \
     --dataset imagenet \

python src/scripts/compute_embeddings.py \
     --detailed_name \
     --fe_type $fe_type \
     --batch_size 1024 \
     --gpu $gpu \
     --model ICL \
     --image_embedding_cache_dir ../image_datasets/wikiart_style/cached_embeddings/ \
     --dataset wikiart-style \

python src/scripts/compute_embeddings.py \
     --detailed_name \
     --fe_type $fe_type \
     --batch_size 1024 \
     --gpu $gpu \
     --model ICL \
     --image_embedding_cache_dir ../image_datasets/wikiart_genre/cached_embeddings/ \
     --dataset wikiart-genre \

python src/scripts/compute_embeddings.py \
     --detailed_name \
     --fe_type $fe_type \
     --batch_size 1024 \
     --gpu $gpu \
     --model ICL \
     --image_embedding_cache_dir ../image_datasets/wikiart_artist/cached_embeddings/ \
     --dataset wikiart-artist \

python src/scripts/compute_embeddings.py \
     --detailed_name \
     --fe_type $fe_type \
     --batch_size 1024 \
     --gpu $gpu \
     --model ICL \
     --image_embedding_cache_dir ../image_datasets/fungi/cached_embeddings/ \
     --dataset fungi \

python src/scripts/compute_embeddings.py \
     --detailed_name \
     --fe_type $fe_type \
     --batch_size 1024 \
     --gpu $gpu \
     --model ICL \
     --image_embedding_cache_dir ../image_datasets/mscoco/cached_embeddings/ \
     --dataset coco \


