from __future__ import annotations

import time
from pathlib import Path

from ..vlm_client import CreateAIClient, _extract_model_payload
from ..vlm_config import REQUEST_PROTOCOL_VERSION, REQUEST_SOURCE, resolve_model, validate_model_identity
from .common import digest, file_hash, read_json, write_json, require, ValidationError
from .prompts import SYSTEM


class StageClient:
    def __init__(self, root, output, options, cache_only=False, reuse_cache_from=()):
        self.output=Path(output); self.options=options; self.cache_only=cache_only
        self.api_identity=resolve_model(options.get("model_name"),options.get("model_provider"),preset=options.get("preset","default"))
        self.model,self.provider=self.api_identity
        self.root=Path(root); self.client=None
        self.calls=0; self.hits=0
        self.reuse_roots=[Path(p).resolve() for p in reuse_cache_from]

    def _client(self):
        if self.client is None:
            import requests
            owner=self
            class CountingSession(requests.Session):
                def post(self,*args,**kwargs):
                    limit=owner.options.get("max_api_calls")
                    require(limit is None or owner.calls<limit,"VLM API call budget exhausted")
                    owner.calls+=1
                    return super().post(*args,**kwargs)
            env={}
            for line in (self.root/".env").read_text(encoding="utf-8-sig").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k,v=line.split("=",1); env[k.strip()]=v.strip().strip('"').strip("'")
            self.client=CreateAIClient(token=env.get("CREATEAI_TOKEN",""),api_url=env.get("CREATEAI_API_URL",""),
                model_name=self.model,model_provider=self.provider,cache_dir=self.output/"api_cache",timeout_s=150,max_retries=1,
                session=CountingSession())
        return self.client

    def _signature(self, image_hash, prompt, model, provider):
        return digest({"image":image_hash,"prompt":prompt,"system":SYSTEM,
            "model":model,"provider":provider,"protocol":REQUEST_PROTOCOL_VERSION,"request_source":REQUEST_SOURCE,
            "max_tokens":self.options.get("max_tokens",6144),"temperature":self.options.get("temperature",0.)})

    def _cached(self, folder, name, signature, identity):
        """(raw path, origin) of a cached answer to this exact request by `identity`, copied into this run, or None."""
        raw_path=folder/(signature+".raw.json")
        if raw_path.exists():
            return raw_path,None
        for prior in self.reuse_roots:
            candidate=prior/"vlm"/name/raw_path.name
            if candidate.exists():
                raw=read_json(candidate)
                validate_model_identity(raw,*identity)
                write_json(raw_path,raw)
                return raw_path,str(candidate)
        return None

    def run(self, name, image_path, prompt, validator, repair_budget=2):
        folder=self.output/"vlm"/name; folder.mkdir(parents=True,exist_ok=True)
        image_hash=file_hash(image_path)
        identity=(self.model,self.provider)
        signature=self._signature(image_hash,prompt,*identity)
        start=time.perf_counter()
        found=self._cached(folder,name,signature,identity)
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
                raw=self._client().query_vision(prompt,Path(image_path).read_bytes(),system_prompt=SYSTEM,mime_type=mime,
                    max_tokens=self.options.get("max_tokens",6144),temperature=self.options.get("temperature",0.))
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
        details=validate_model_identity(raw,*identity)
        parsed=_extract_model_payload(raw)
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
