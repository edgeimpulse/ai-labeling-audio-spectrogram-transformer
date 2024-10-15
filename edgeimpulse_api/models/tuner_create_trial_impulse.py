# coding: utf-8

"""
    Edge Impulse API

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Generated by: https://openapi-generator.tech
"""


from __future__ import annotations
from inspect import getfullargspec
import pprint
import re  # noqa: F401
import json


from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, StrictStr

class TunerCreateTrialImpulse(BaseModel):
    id: Optional[StrictStr] = None
    experiment: Optional[StrictStr] = None
    original_trial_id: Optional[StrictStr] = None
    input_blocks: Optional[List[Dict[str, Any]]] = Field(None, alias="inputBlocks")
    dsp_blocks: Optional[List[Dict[str, Any]]] = Field(None, alias="dspBlocks")
    learn_blocks: Optional[List[Dict[str, Any]]] = Field(None, alias="learnBlocks")
    __properties = ["id", "experiment", "original_trial_id", "inputBlocks", "dspBlocks", "learnBlocks"]

    class Config:
        allow_population_by_field_name = True
        validate_assignment = False

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> TunerCreateTrialImpulse:
        """Create an instance of TunerCreateTrialImpulse from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> TunerCreateTrialImpulse:
        """Create an instance of TunerCreateTrialImpulse from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return TunerCreateTrialImpulse.construct(**obj)

        _obj = TunerCreateTrialImpulse.construct(**{
            "id": obj.get("id"),
            "experiment": obj.get("experiment"),
            "original_trial_id": obj.get("original_trial_id"),
            "input_blocks": obj.get("inputBlocks"),
            "dsp_blocks": obj.get("dspBlocks"),
            "learn_blocks": obj.get("learnBlocks")
        })
        return _obj
