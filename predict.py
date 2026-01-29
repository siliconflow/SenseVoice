# Cog预测入口文件 - 用于模型推理和在线API服务
import os
import sys
import time
import threading
import torch
import logging
from typing import Optional
from cog import BasePredictor, Path, Input

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ["SENSEVOICE_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Predictor(BasePredictor):
    """SenseVoice 预测器类（单例模式，模型全局共享）"""

    _model = None
    _model_lock = threading.Lock()

    def setup(self):
        """加载模型（仅第一次调用时加载）"""
        if Predictor._model is not None:
            logger.info("模型已存在，跳过重复加载")
            return

        with Predictor._model_lock:
            # 双重检查
            if Predictor._model is not None:
                return

            logger.info("正在加载 SenseVoice 模型...")
            self.device = os.environ.get("SENSEVOICE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"使用设备: {self.device}")

            from funasr import AutoModel

            Predictor._model = AutoModel(
                model="iic/SenseVoiceSmall",
                trust_remote_code=True,
                device=self.device,
            )

            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

            logger.info("SenseVoice 模型加载完成（全局单例）")

    def predict(
        self,
        input_audio: Path,
        language: str = "auto",
        use_itn: bool = True,
        batch_size: int = 1,
    ) -> str:
        """
        音频转写预测

        Args:
            input_audio: 输入音频文件路径
            language: 语言选项 (auto, zh, en, yue, ja, ko)
            use_itn: 是否使用逆文本正规化
            batch_size: 批处理大小

        Returns:
            转写结果文本
        """
        start_time = time.time()

        # 确保模型已加载
        if Predictor._model is None:
            self.setup()

        logger.info(f"开始处理音频: {input_audio}")
        logger.info(f"语言选项: {language}, ITN: {use_itn}")

        try:
            # 使用全局共享模型
            result = Predictor._model.generate(
                input=input_audio,
                cache={},
                language=language,
                use_itn=1 if use_itn else 0,
                batch_size=batch_size,
            )

            processing_time = time.time() - start_time

            # 解析结果
            if result and len(result) > 0:
                if isinstance(result[0], dict) and 'text' in result[0]:
                    text = result[0]['text']
                    logger.info(f"转写完成，耗时: {processing_time:.2f}s")
                    return text.strip()
                else:
                    logger.info(f"转写完成，耗时: {processing_time:.2f}s")
                    return str(result[0]).strip()

            logger.warning("转写结果为空")
            return ""

        except Exception as e:
            logger.error(f"转写失败: {str(e)}", exc_info=True)
            raise


# 全局模型实例（用于 API 服务复用）
_transcribe_model = None
_model_lock = threading.Lock()


def _get_model():
    """获取或创建全局模型单例"""
    global _transcribe_model
    if _transcribe_model is None:
        with _model_lock:
            if _transcribe_model is not None:
                return _transcribe_model
            logger.info("创建 API 服务全局模型单例...")
            device = os.environ.get("SENSEVOICE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            from funasr import AutoModel
            _transcribe_model = AutoModel(
                model="iic/SenseVoiceSmall",
                trust_remote_code=True,
                device=device,
            )
            logger.info(f"API 服务全局模型创建完成，使用设备: {device}")
    return _transcribe_model


def transcribe_audio(
    audio_path: str,
    language: str = "auto",
    use_itn: bool = True,
) -> dict:
    """
    音频转写函数（API服务调用，使用全局模型单例）

    Args:
        audio_path: 音频文件路径或URL
        language: 语言选项
        use_itn: 是否使用逆文本正规化

    Returns:
        包含转写结果的字典
    """
    start_time = time.time()

    logger.info(f"API转写请求: {audio_path}")

    try:
        # 使用全局模型，避免重复加载
        model = _get_model()

        result = model.generate(
            input=audio_path,
            cache={},
            language=language,
            use_itn=1 if use_itn else 0,
        )

        processing_time = time.time() - start_time

        if result and len(result) > 0:
            if isinstance(result[0], dict):
                return {
                    "success": True,
                    "text": result[0].get('text', '').strip(),
                    "time": processing_time,
                }
            return {
                "success": True,
                "text": str(result[0]).strip(),
                "time": processing_time,
            }

        return {
            "success": False,
            "error": "转写结果为空",
            "time": processing_time,
        }

    except Exception as e:
        logger.error(f"API转写失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "time": time.time() - start_time,
        }