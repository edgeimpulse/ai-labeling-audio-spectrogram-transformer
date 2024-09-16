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
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr
from edgeimpulse_api.models.classify_sample_response_classification import ClassifySampleResponseClassification
from edgeimpulse_api.models.raw_sample_data import RawSampleData

class ClassifySampleResponseAllOf(BaseModel):
    classifications: List[ClassifySampleResponseClassification] = ...
    sample: RawSampleData = ...
    window_size_ms: StrictInt = Field(..., alias="windowSizeMs", description="Size of the sliding window (as set by the impulse) in milliseconds.")
    window_increase_ms: StrictInt = Field(..., alias="windowIncreaseMs", description="Number of milliseconds that the sliding window increased with (as set by the impulse)")
    already_in_database: StrictBool = Field(..., alias="alreadyInDatabase", description="Whether this sample is already in the training database")
    warning: Optional[StrictStr] = None
    __properties = ["classifications", "sample", "windowSizeMs", "windowIncreaseMs", "alreadyInDatabase", "warning"]

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
    def from_json(cls, json_str: str) -> ClassifySampleResponseAllOf:
        """Create an instance of ClassifySampleResponseAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in classifications (list)
        _items = []
        if self.classifications:
            for _item in self.classifications:
                if _item:
                    _items.append(_item.to_dict())
            _dict['classifications'] = _items
        # override the default output from pydantic by calling `to_dict()` of sample
        if self.sample:
            _dict['sample'] = self.sample.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ClassifySampleResponseAllOf:
        """Create an instance of ClassifySampleResponseAllOf from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return ClassifySampleResponseAllOf.construct(**obj)

        _obj = ClassifySampleResponseAllOf.construct(**{
            "classifications": [ClassifySampleResponseClassification.from_dict(_item) for _item in obj.get("classifications")] if obj.get("classifications") is not None else None,
            "sample": RawSampleData.from_dict(obj.get("sample")) if obj.get("sample") is not None else None,
            "window_size_ms": obj.get("windowSizeMs"),
            "window_increase_ms": obj.get("windowIncreaseMs"),
            "already_in_database": obj.get("alreadyInDatabase"),
            "warning": obj.get("warning")
        })
        return _obj
