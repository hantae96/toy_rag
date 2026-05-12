import logging

from app.middleware.request_log_middleware import RequestIdFilter


class LoggingConfigurator:
    FORMAT = "[%(levelname)s] %(name)s [%(request_id)s] - %(message)s - %(asctime)s"

    @classmethod
    def configure(cls) -> None:
        logging.basicConfig(level=logging.INFO, format=cls.FORMAT)
        request_id_filter = RequestIdFilter()
        root_logger = logging.getLogger()
        root_logger.addFilter(request_id_filter)
        for handler in root_logger.handlers:
            handler.addFilter(request_id_filter)
