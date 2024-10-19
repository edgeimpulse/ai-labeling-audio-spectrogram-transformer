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
from edgeimpulse_api.models.dsp_run_graph import DspRunGraph
from edgeimpulse_api.models.dsp_run_response_all_of_performance import DspRunResponseAllOfPerformance
from edgeimpulse_api.models.raw_sample_data import RawSampleData

class DspRunResponseWithSampleAllOf(BaseModel):
    features: List[float] = Field(..., description="Array of processed features. Laid out according to the names in 'labels'")
    graphs: List[DspRunGraph] = Field(..., description="Graphs to plot to give an insight in how the DSP process ran")
    labels: Optional[List[StrictStr]] = Field(None, description="Labels of the feature axes")
    state_string: Optional[StrictStr] = Field(None, description="String representation of the DSP state returned")
    label_at_end_of_window: Optional[StrictStr] = Field(None, alias="labelAtEndOfWindow", description="Label for the window (only present for time-series data)")
    sample: RawSampleData = ...
    performance: Optional[DspRunResponseAllOfPerformance] = None
    can_profile_performance: StrictBool = Field(..., alias="canProfilePerformance")
    __properties = ["features", "graphs", "labels", "state_string", "labelAtEndOfWindow", "sample", "performance", "canProfilePerformance"]

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
    def from_json(cls, json_str: str) -> DspRunResponseWithSampleAllOf:
        """Create an instance of DspRunResponseWithSampleAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in graphs (list)
        _items = []
        if self.graphs:
            for _item in self.graphs:
                if _item:
                    _items.append(_item.to_dict())
            _dict['graphs'] = _items
        # override the default output from pydantic by calling `to_dict()` of sample
        if self.sample:
            _dict['sample'] = self.sample.to_dict()
        # override the default output from pydantic by calling `to_dict()` of performance
        if self.performance:
            _dict['performance'] = self.performance.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DspRunResponseWithSampleAllOf:
        """Create an instance of DspRunResponseWithSampleAllOf from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DspRunResponseWithSampleAllOf.construct(**obj)

        _obj = DspRunResponseWithSampleAllOf.construct(**{
            "features": obj.get("features"),
            "graphs": [DspRunGraph.from_dict(_item) for _item in obj.get("graphs")] if obj.get("graphs") is not None else None,
            "labels": obj.get("labels"),
            "state_string": obj.get("state_string"),
            "label_at_end_of_window": obj.get("labelAtEndOfWindow"),
            "sample": RawSampleData.from_dict(obj.get("sample")) if obj.get("sample") is not None else None,
            "performance": DspRunResponseAllOfPerformance.from_dict(obj.get("performance")) if obj.get("performance") is not None else None,
            "can_profile_performance": obj.get("canProfilePerformance")
        })
        return _obj

