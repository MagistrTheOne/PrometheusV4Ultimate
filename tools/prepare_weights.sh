#!/bin/bash
# Prepare model weights for PrometheusULTIMATE v4
# This script provides instructions for downloading weights from HuggingFace

set -e

echo "🚀 PrometheusULTIMATE v4 - Model Weights Preparation"
echo "=================================================="

# Create weights directory structure
mkdir -p weights/radon weights/oracle

echo "📁 Created weights directory structure"

# Model mapping
declare -A models=(
    ["MagistrTheOne/RadonSAI-Small"]="weights/radon/small-0.1b"
    ["MagistrTheOne/RadonSAI"]="weights/radon/base-0.8b"
    ["MagistrTheOne/RadonSAI-Balanced"]="weights/radon/balanced-3b"
    ["MagistrTheOne/RadonSAI-Efficient"]="weights/radon/efficient-3b"
    ["MagistrTheOne/RadonSAI-Pretrained"]="weights/radon/pretrained-7b"
    ["MagistrTheOne/RadonSAI-Ultra"]="weights/radon/ultra-13b"
    ["MagistrTheOne/RadonSAI-Mega"]="weights/radon/mega-70b"
    ["MagistrTheOne/RadonSAI-GPT5Competitor"]="weights/radon/gpt5competitor"
    ["MagistrTheOne/RadonSAI-DarkUltima"]="weights/radon/darkultima"
    ["MagistrTheOne/oracle850b-moe"]="weights/oracle/moe-850b"
)

echo ""
echo "📋 Model Download Instructions:"
echo "==============================="

for hf_model in "${!models[@]}"; do
    local_path="${models[$hf_model]}"
    echo ""
    echo "🔹 $hf_model"
    echo "   Local path: $local_path"
    echo "   Command:"
    echo "   git clone https://huggingface.co/$hf_model $local_path"
    echo "   # Or using huggingface-hub:"
    echo "   python -c \"from huggingface_hub import snapshot_download; snapshot_download('$hf_model', local_dir='$local_path')\""
done

echo ""
echo "🔧 Serving Configuration:"
echo "========================"
echo ""
echo "For vLLM serving (GPU):"
echo "python -m vllm.entrypoints.openai.api_server \\"
echo "  --model ./weights/radon/balanced-3b \\"
echo "  --dtype float16 \\"
echo "  --max-model-len 4096 \\"
echo "  --tensor-parallel-size 1 \\"
echo "  --host 0.0.0.0 --port 9001"
echo ""
echo "For llama.cpp serving (CPU):"
echo "./server -m ./weights/radon/small-0.1b/model.gguf -c 4096 -ngl 0 -t 8 -p 9002"
echo ""
echo "📝 Notes:"
echo "- Models are ready for download from HuggingFace"
echo "- Use the provided HF token for authentication"
echo "- Weights will be stored locally in ./weights/"
echo "- Update docker-compose.yml to mount weights directory"
echo "- Configure serving endpoints in libs/common/llm_providers.py"
echo ""
echo "✅ Preparation complete! Run the download commands above when ready."
