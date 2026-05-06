import logging
import time
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request : Request, call_next):
        start_time = time.time()

        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(f"[{request.method}] {response.status_code} {request.url.path} {process_time:.3f}s ")
            return response
       
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"unexpected error : {str(e)} ({process_time:.3f}s)")
            return JSONResponse(
                status_code=500,
                content={"success" : False, "message" : "서버 내부 오류가 발생했습니다."}
            )