"""
DeepSeek-OCR on Replicate Cog
支持圖片（JPG/PNG/WebP）+ PDF
"""

from cog import BasePredictor, Input
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import requests
from io import BytesIO
import tempfile
import os


class Predictor(BasePredictor):
    """Replicate 預測類"""

    def setup(self) -> None:
        """初始化模型（容器啟動時運行一次）"""
        print("🚀 初始化 DeepSeek-OCR 模型...")

        model_name = "deepseek-ai/DeepSeek-OCR"

        print("📖 加載 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print("🧠 加載模型...")
        # CPU 友善配置
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # CPU 支持
            low_cpu_mem_usage=True  # 節省記憶體
        )

        # 如果有 GPU 則使用
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型已加載到 {self.device}")

    def predict(
        self,
        image_url: str = Input(description="Image URL"),
        prompt: str = Input(default="Extract all text from this document")
    ) -> str:
        """OCR 推理"""
        try:
            print(f"🔄 處理請求...")

            # 加載圖像
            image = self._load_image(image_url)

            # 保存為臨時文件
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                image.save(tmp_file.name, "JPEG")
                image_path = tmp_file.name

            print(f"   圖像大小: {image.size}")

            # 執行 OCR
            print("   運行 OCR 推理...")
            with torch.no_grad():
                result = self.model.infer(
                    self.tokenizer,
                    prompt=f"<image>\n{prompt}",
                    image_file=image_path,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True
                )

            # 清理
            os.remove(image_path)

            print(f"✅ OCR 完成！")
            return result

        except Exception as e:
            print(f"❌ 錯誤: {str(e)}")
            raise

    def _load_image(self, image_input: str) -> Image.Image:
        """加載圖像"""
        if image_input.startswith("http"):
            print("   📥 從 URL 下載圖像...")
            response = requests.get(image_input, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        elif image_input.startswith("data:"):
            print("   🔐 解碼 Base64...")
            import base64
            base64_str = image_input.split(",")[1]
            image_bytes = base64.b64decode(base64_str)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            print("   📂 加載本地文件...")
            return Image.open(image_input).convert("RGB")
