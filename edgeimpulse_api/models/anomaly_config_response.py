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
from edgeimpulse_api.models.anomaly_config_axes_inner import AnomalyConfigAxesInner
from edgeimpulse_api.models.dependency_data import DependencyData

class AnomalyConfigResponse(BaseModel):
    success: StrictBool = Field(..., description="Whether the operation succeeded")
    error: Optional[StrictStr] = Field(None, description="Optional error description (set if 'success' was false)")
    dependencies: DependencyData = ...
    name: StrictStr = ...
    axes: List[AnomalyConfigAxesInner] = Field(..., description="Selectable axes for the anomaly detection block")
    trained: StrictBool = Field(..., description="Whether the block is trained")
    cluster_count: Optional[StrictInt] = Field(None, alias="clusterCount", description="Number of clusters for K-means, or number of components for GMM (in config)")
    selected_axes: List[StrictInt] = Field(..., alias="selectedAxes", description="Selected clusters (in config)")
    minimum_confidence_rating: float = Field(..., alias="minimumConfidenceRating", description="Minimum confidence rating for this block, scores above this number will be flagged as anomaly.")
    __properties = ["success", "error", "dependencies", "name", "axes", "trained", "clusterCount", "selectedAxes", "minimumConfidenceRating"]

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
    def from_json(cls, json_str: str) -> AnomalyConfigResponse:
        """Create an instance of AnomalyConfigResponse from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of dependencies
        if self.dependencies:
            _dict['dependencies'] = self.dependencies.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each item in axes (list)
        _items = []
        if self.axes:
            for _item in self.axes:
                if _item:
                    _items.append(_item.to_dict())
            _dict['axes'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> AnomalyConfigResponse:
        """Create an instance of AnomalyConfigResponse from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return AnomalyConfigResponse.construct(**obj)

        _obj = AnomalyConfigResponse.construct(**{
            "success": obj.get("success"),
            "error": obj.get("error"),
            "dependencies": DependencyData.from_dict(obj.get("dependencies")) if obj.get("dependencies") is not None else None,
            "name": obj.get("name"),
            "axes": [AnomalyConfigAxesInner.from_dict(_item) for _item in obj.get("axes")] if obj.get("axes") is not None else None,
            "trained": obj.get("trained"),
            "cluster_count": obj.get("clusterCount"),
            "selected_axes": obj.get("selectedAxes"),
            "minimum_confidence_rating": obj.get("minimumConfidenceRating")
        })
        return _obj

