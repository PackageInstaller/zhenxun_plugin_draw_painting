from nonebot import get_driver

from zhenxun.services.log import logger

from ..config import ensure_directories
from ..services.image_features import image_feature_service

driver = get_driver()


@driver.on_startup
async def initialize() -> None:
    """Start model preparation and the non-blocking image feature index."""

    ensure_directories()
    image_feature_service.start()
    logger.info("立绘特征索引与实时监控已在后台启动")


@driver.on_shutdown
async def shutdown() -> None:
    await image_feature_service.stop()
