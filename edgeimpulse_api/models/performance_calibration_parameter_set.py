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
from pydantic import BaseModel, Field, StrictBool, StrictStr
from edgeimpulse_api.models.performance_calibration_detection import PerformanceCalibrationDetection
from edgeimpulse_api.models.performance_calibration_parameter_set_aggregate_stats import PerformanceCalibrationParameterSetAggregateStats
from edgeimpulse_api.models.performance_calibration_parameter_set_stats_inner import PerformanceCalibrationParameterSetStatsInner
from edgeimpulse_api.models.performance_calibration_parameters import PerformanceCalibrationParameters

class PerformanceCalibrationParameterSet(BaseModel):
    detections: List[PerformanceCalibrationDetection] = Field(..., description="All of the detections using this parameter set")
    is_best: StrictBool = Field(..., alias="isBest", description="Whether this is considered the best parameter set")
    labels: List[StrictStr] = Field(..., description="All of the possible labels in the detections array")
    aggregate_stats: PerformanceCalibrationParameterSetAggregateStats = Field(..., alias="aggregateStats")
    stats: List[PerformanceCalibrationParameterSetStatsInner] = ...
    params: PerformanceCalibrationParameters = ...
    window_size_ms: float = Field(..., alias="windowSizeMs", description="The size of the input block window in milliseconds.")
    __properties = ["detections", "isBest", "labels", "aggregateStats", "stats", "params", "windowSizeMs"]

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
    def from_json(cls, json_str: str) -> PerformanceCalibrationParameterSet:
        """Create an instance of PerformanceCalibrationParameterSet from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in detections (list)
        _items = []
        if self.detections:
            for _item in self.detections:
                if _item:
                    _items.append(_item.to_dict())
            _dict['detections'] = _items
        # override the default output from pydantic by calling `to_dict()` of aggregate_stats
        if self.aggregate_stats:
            _dict['aggregateStats'] = self.aggregate_stats.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each item in stats (list)
        _items = []
        if self.stats:
            for _item in self.stats:
                if _item:
                    _items.append(_item.to_dict())
            _dict['stats'] = _items
        # override the default output from pydantic by calling `to_dict()` of params
        if self.params:
            _dict['params'] = self.params.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> PerformanceCalibrationParameterSet:
        """Create an instance of PerformanceCalibrationParameterSet from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return PerformanceCalibrationParameterSet.construct(**obj)

        _obj = PerformanceCalibrationParameterSet.construct(**{
            "detections": [PerformanceCalibrationDetection.from_dict(_item) for _item in obj.get("detections")] if obj.get("detections") is not None else None,
            "is_best": obj.get("isBest"),
            "labels": obj.get("labels"),
            "aggregate_stats": PerformanceCalibrationParameterSetAggregateStats.from_dict(obj.get("aggregateStats")) if obj.get("aggregateStats") is not None else None,
            "stats": [PerformanceCalibrationParameterSetStatsInner.from_dict(_item) for _item in obj.get("stats")] if obj.get("stats") is not None else None,
            "params": PerformanceCalibrationParameters.from_dict(obj.get("params")) if obj.get("params") is not None else None,
            "window_size_ms": obj.get("windowSizeMs")
        })
        return _obj

