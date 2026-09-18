from typing import Optional

from pydantic import BaseModel

_plugin_config: Optional["Config"] = None

class Config(BaseModel):
    # OSCA OSS（S3 兼容）
    realcugan_oss_endpoint: str = "https://fgws3-ocloud.ihep.ac.cn"
    realcugan_oss_access_key: str = ""
    realcugan_oss_secret_key: str = ""
    realcugan_oss_bucket: str = ""
    realcugan_oss_prefix: str = "temps/"
    # 预签名下载链接有效期（秒），默认 1 天
    realcugan_oss_url_expires: int = 86400
    # 云端对象保留时间（秒），到期后自动删除
    realcugan_oss_retain_seconds: int = 86400

class _LazyConfig:
    def _get(self) -> Config:
        global _plugin_config
        if _plugin_config is None:
            try:
                from nonebot import get_plugin_config

                _plugin_config = get_plugin_config(Config)
            except Exception:
                _plugin_config = Config()
        return _plugin_config

    def __getattr__(self, item: str):
        return getattr(self._get(), item)


plugin_config = _LazyConfig()
