from __future__ import annotations

import os
import time
from pathlib import Path

from .. import vision_client
from .. import vlm_bridge
from .common import digest, file_hash, read_json, write_json, require, ValidationError
from .prompts import SYSTEM

# The model answers through an OpenAI-compatible API (vision_client). Its settings (OPENAI_API_KEY, OPENAI_BASE_URL,
# OPENAI_MODEL) come from .env in the project folder or its parent, and the process environment. With
# PAVE_VLM_BACKEND=bridge the requests go to a file queue instead, answered by agents (vlm_bridge).


class StageClient:
    def __init__(self, root, output, options, cache_only=False, reuse_cache_from=()):
        self.output=Path(output); self.options=options; self.cache_only=cache_only
        self.root=Path(root); self.client=None
        self.env=vision_client.environment(self.root/".env",self.root.parent/".env")
        # PAVE_VLM_BACKEND=bridge: requests go to the agent queue (vlm_bridge) under the bridge's own model identity
        self.backend=vlm_bridge.backend()
        try:
            self.api_identity=self._api_identity()
        except vision_client.VisionError:
            if self.backend!="bridge": raise
            self.api_identity=None  # no API model configured: the queue has no API answers to reuse
        self.model,self.provider=vlm_bridge.identity() if self.backend=="bridge" else self.api_identity
        self.calls=0; self.hits=0
        # seconds to wait for one answer; long answers (large output limits) may need more than the default
        self.timeout_s=int(os.environ.get("PAVE_VLM_TIMEOUT_S") or 150)
        # options["system_prompt"] replaces the harness system prompt; None sends none
        self.system=options.get("system_prompt",SYSTEM)
        self.reuse_roots=[Path(p).resolve() for p in reuse_cache_from]

    # What depends on the API: the model's identity, the protocol in the cache key, and the client.
    def _api_identity(self):
        s=vision_client.settings(self.env,need_key=False)
        return vision_client.identity(s["model"],s["base_url"])

    def _api_protocol(self):
        return {"protocol":vision_client.PROTOCOL,"request_source":"chat_completions"}

    def _api_client(self, count):
        s=vision_client.settings(self.env)
        return vision_client.VisionClient(s["api_key"],s["model"],s["base_url"],timeout_s=self.timeout_s,max_retries=1,before_request=count)

    def _client(self):
        if self.client is None:
            def count():  # every HTTP request, retries included, counts against max_api_calls
                limit=self.options.get("max_api_calls")
                require(limit is None or self.calls<limit,"VLM API call budget exhausted")
                self.calls+=1
            self.client=self._api_client(count)
        return self.client

    def max_tokens(self, name):
        """The output limit of stage `name`: options["stage_max_tokens"] maps a stage-name prefix (e.g. "usage", which
        also covers usage_detail_* and *_repair1) to a limit, the longest matching prefix winning; else max_tokens."""
        override=self.options.get("stage_max_tokens") or {}
        require(isinstance(override,dict) and all(isinstance(v,int) and v>0 for v in override.values()),
                "stage_max_tokens must map stage-name prefixes to positive integers")
        matches=[k for k in override if name.startswith(k)]
        return override[max(matches,key=len)] if matches else self.options.get("max_tokens",6144)

    def _signature(self, image_hash, prompt, model, provider, name=""):
        return digest({"image":image_hash,"prompt":prompt,"system":self.system,"model":model,"provider":provider,**self._api_protocol(),
            "max_tokens":self.max_tokens(name),"temperature":self.options.get("temperature",0.)})

    def _cached(self, folder, name, signature, identity):
        """(raw path, origin) of a cached answer to this exact request by `identity`, copied into this run, or None."""
        raw_path=folder/(signature+".raw.json")
        if raw_path.exists():
            return raw_path,None
        for prior in self.reuse_roots:
            candidate=prior/"vlm"/name/raw_path.name
            if candidate.exists():
                raw=read_json(candidate)
                vision_client.validate_identity(raw,*identity)
                write_json(raw_path,raw)
                return raw_path,str(candidate)
        return None

    def run(self, name, image_path, prompt, validator, repair_budget=2):
        folder=self.output/"vlm"/name; folder.mkdir(parents=True,exist_ok=True)
        image_hash=file_hash(image_path)
        identity=(self.model,self.provider)
        signature=self._signature(image_hash,prompt,*identity,name=name)
        start=time.perf_counter()
        found=self._cached(folder,name,signature,identity)
        if found is None and self.backend=="bridge" and self.api_identity and vlm_bridge.reuse_identical():
            # A request identical in image and prompt to one the API model already answered keeps that answer;
            # latest.json records which model gave it.
            api_signature=self._signature(image_hash,prompt,*self.api_identity,name=name)
            found=self._cached(folder,name,api_signature,self.api_identity)
            if found is not None:
                identity,signature=self.api_identity,api_signature
        raw_path,cache_origin=found if found is not None else (folder/(signature+".raw.json"),None)
        (folder/(signature+".prompt.txt")).write_text(prompt,encoding="utf-8")
        if raw_path.exists():
            raw=read_json(raw_path); self.hits+=1; cached=True
        else:
            require(not self.cache_only,f"Missing cached VLM stage {name}")
            print(f"VLM {name}: requesting {self.provider}/{self.model}",flush=True)
            before=self.calls
            mime="image/jpeg" if Path(image_path).suffix.lower() in (".jpg",".jpeg") else "image/png"
            try:
                if self.backend=="bridge":
                    limit=self.options.get("max_api_calls")
                    require(limit is None or self.calls<limit,"VLM API call budget exhausted")
                    self.calls+=1
                    raw=vlm_bridge.ask(name,image_path,prompt,self.system or "",signature,self.output)
                else:
                    raw=self._client().query_vision(prompt,Path(image_path).read_bytes(),system_prompt=self.system,mime_type=mime,
                        max_tokens=self.max_tokens(name),temperature=self.options.get("temperature",0.))
            except Exception as error:
                import re
                message=str(error)
                match=re.search(r"\b(?:vlm|HTTP)\s+(\d{3})\b",message,re.I)
                write_json(folder/"request_error.json",{"error_type":type(error).__name__,
                    "http_status":int(match.group(1)) if match else None,"image_bytes":Path(image_path).stat().st_size,
                    "timeout":bool(re.search(r"timeout|timed out",message,re.I)),"response_body_logged":False})
                raise
            cached=self.calls==before
            if cached: self.hits+=1
            write_json(raw_path,raw)
        details=vision_client.validate_identity(raw,*identity)
        parsed=vision_client.extract_payload(raw)
        write_json(folder/(signature+".parsed.json"),parsed)
        repair_stage=None
        try:
            validated=validator(parsed)
        except ValidationError as error:
            write_json(folder/(signature+".validation_error.json"),{"error":str(error)})
            if repair_budget<=0: raise
            import json
            corrected_prompt=prompt+"\n\nThe previous response failed a structural check: "+str(error)+"\nReturn a corrected complete JSON response. Do not fabricate evidence to satisfy the check; unresolved decisions may use empty pairs.\nPrevious response:\n"+json.dumps(parsed,ensure_ascii=False)
            repair_stage=name+"_repair1"
            validated=self.run(repair_stage,image_path,corrected_prompt,validator,repair_budget-1)
        write_json(folder/(signature+".validated.json"),validated)
        write_json(folder/"latest.json",{"signature":signature,"cached":cached,"elapsed_s":round(time.perf_counter()-start,3),
            "actual_model":details,"raw_file":raw_path.name,"parsed_file":signature+".parsed.json",
            "validated_file":signature+".validated.json","repair_stage":repair_stage,"cache_origin":cache_origin})
        print(f"VLM {name}: {'cached' if cached else 'completed'}",flush=True)
        return validated
