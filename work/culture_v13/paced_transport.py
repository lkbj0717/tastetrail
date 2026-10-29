"""Bounded global request rate for a shared proxy; no answer-dependent retries."""
import threading
import time
import httpx

_LOCK=threading.Lock()
_NEXT=0.


class PacedTransport(httpx.BaseTransport):
    def __init__(self, interval=3.):
        self.interval=float(interval)
        self.inner=httpx.HTTPTransport()
        self.events=[]

    def handle_request(self,request):
        global _NEXT
        with _LOCK:
            delay=max(0.,_NEXT-time.monotonic())
            if delay:time.sleep(delay)
            _NEXT=time.monotonic()+self.interval
        started=time.monotonic()
        response=self.inner.handle_request(request)
        event={'status':response.status_code,'start_monotonic':started,
               'interval_seconds':self.interval}
        if response.status_code>=400:
            # Only retain a short machine classification, never headers,
            # request text, credentials or the full server error message.
            try:
                data=response.read();import json
                error=json.loads(data).get('error',{})
                if isinstance(error,dict):
                    event['error_type']=str(error.get('type',''))[:100]
                    event['error_code']=str(error.get('code',''))[:100]
            except (ValueError,TypeError):pass
        self.events.append(event)
        return response

    def close(self):
        self.inner.close()
