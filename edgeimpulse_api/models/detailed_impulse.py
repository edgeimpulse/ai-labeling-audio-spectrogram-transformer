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


from typing import List, Optional
from pydantic import BaseModel, Field, StrictBool, StrictStr
from edgeimpulse_api.models.detailed_impulse_dsp_block_configs_inner import DetailedImpulseDspBlockConfigsInner
from edgeimpulse_api.models.detailed_impulse_learn_block_anomaly_configs_inner import DetailedImpulseLearnBlockAnomalyConfigsInner
from edgeimpulse_api.models.detailed_impulse_learn_block_keras_configs_inner import DetailedImpulseLearnBlockKerasConfigsInner
from edgeimpulse_api.models.detailed_impulse_metric import DetailedImpulseMetric
from edgeimpulse_api.models.detailed_impulse_pretrained_model_info import DetailedImpulsePretrainedModelInfo
from edgeimpulse_api.models.impulse import Impulse

class DetailedImpulse(BaseModel):
    impulse: Impulse = ...
    metrics: List[DetailedImpulseMetric] = ...
    dsp_block_configs: List[DetailedImpulseDspBlockConfigsInner] = Field(..., alias="dspBlockConfigs")
    learn_block_keras_configs: List[DetailedImpulseLearnBlockKerasConfigsInner] = Field(..., alias="learnBlockKerasConfigs")
    learn_block_anomaly_configs: List[DetailedImpulseLearnBlockAnomalyConfigsInner] = Field(..., alias="learnBlockAnomalyConfigs")
    pretrained_model_info: Optional[DetailedImpulsePretrainedModelInfo] = Field(None, alias="pretrainedModelInfo")
    is_stale: StrictBool = Field(..., alias="isStale", description="Whether this impulse contains blocks with \"stale\" features (i.e. the dataset has changed since features were generated)")
    tags: List[StrictStr] = Field(..., description="Tags associated with this impulse")
    __properties = ["impulse", "metrics", "dspBlockConfigs", "learnBlockKerasConfigs", "learnBlockAnomalyConfigs", "pretrainedModelInfo", "isStale", "tags"]

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
    def from_json(cls, json_str: str) -> DetailedImpulse:
        """Create an instance of DetailedImpulse from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of impulse
        if self.impulse:
            _dict['impulse'] = self.impulse.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each item in metrics (list)
        _items = []
        if self.metrics:
            for _item in self.metrics:
                if _item:
                    _items.append(_item.to_dict())
            _dict['metrics'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in dsp_block_configs (list)
        _items = []
        if self.dsp_block_configs:
            for _item in self.dsp_block_configs:
                if _item:
                    _items.append(_item.to_dict())
            _dict['dspBlockConfigs'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in learn_block_keras_configs (list)
        _items = []
        if self.learn_block_keras_configs:
            for _item in self.learn_block_keras_configs:
                if _item:
                    _items.append(_item.to_dict())
            _dict['learnBlockKerasConfigs'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in learn_block_anomaly_configs (list)
        _items = []
        if self.learn_block_anomaly_configs:
            for _item in self.learn_block_anomaly_configs:
                if _item:
                    _items.append(_item.to_dict())
            _dict['learnBlockAnomalyConfigs'] = _items
        # override the default output from pydantic by calling `to_dict()` of pretrained_model_info
        if self.pretrained_model_info:
            _dict['pretrainedModelInfo'] = self.pretrained_model_info.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DetailedImpulse:
        """Create an instance of DetailedImpulse from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DetailedImpulse.construct(**obj)

        _obj = DetailedImpulse.construct(**{
            "impulse": Impulse.from_dict(obj.get("impulse")) if obj.get("impulse") is not None else None,
            "metrics": [DetailedImpulseMetric.from_dict(_item) for _item in obj.get("metrics")] if obj.get("metrics") is not None else None,
            "dsp_block_configs": [DetailedImpulseDspBlockConfigsInner.from_dict(_item) for _item in obj.get("dspBlockConfigs")] if obj.get("dspBlockConfigs") is not None else None,
            "learn_block_keras_configs": [DetailedImpulseLearnBlockKerasConfigsInner.from_dict(_item) for _item in obj.get("learnBlockKerasConfigs")] if obj.get("learnBlockKerasConfigs") is not None else None,
            "learn_block_anomaly_configs": [DetailedImpulseLearnBlockAnomalyConfigsInner.from_dict(_item) for _item in obj.get("learnBlockAnomalyConfigs")] if obj.get("learnBlockAnomalyConfigs") is not None else None,
            "pretrained_model_info": DetailedImpulsePretrainedModelInfo.from_dict(obj.get("pretrainedModelInfo")) if obj.get("pretrainedModelInfo") is not None else None,
            "is_stale": obj.get("isStale"),
            "tags": obj.get("tags")
        })
        return _obj
