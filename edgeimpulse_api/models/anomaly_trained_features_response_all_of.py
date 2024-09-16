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


from typing import List
from pydantic import BaseModel, Field, StrictInt
from edgeimpulse_api.models.anomaly_trained_features_response_all_of_data import AnomalyTrainedFeaturesResponseAllOfData

class AnomalyTrainedFeaturesResponseAllOf(BaseModel):
    total_sample_count: StrictInt = Field(..., alias="totalSampleCount", description="Total number of windows in the data set")
    data: List[AnomalyTrainedFeaturesResponseAllOfData] = ...
    __properties = ["totalSampleCount", "data"]

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
    def from_json(cls, json_str: str) -> AnomalyTrainedFeaturesResponseAllOf:
        """Create an instance of AnomalyTrainedFeaturesResponseAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in data (list)
        _items = []
        if self.data:
            for _item in self.data:
                if _item:
                    _items.append(_item.to_dict())
            _dict['data'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> AnomalyTrainedFeaturesResponseAllOf:
        """Create an instance of AnomalyTrainedFeaturesResponseAllOf from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return AnomalyTrainedFeaturesResponseAllOf.construct(**obj)

        _obj = AnomalyTrainedFeaturesResponseAllOf.construct(**{
            "total_sample_count": obj.get("totalSampleCount"),
            "data": [AnomalyTrainedFeaturesResponseAllOfData.from_dict(_item) for _item in obj.get("data")] if obj.get("data") is not None else None
        })
        return _obj

