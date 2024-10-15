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


from typing import Optional
from pydantic import BaseModel, Field, StrictBool, StrictInt
from edgeimpulse_api.models.keras_visual_layer_type import KerasVisualLayerType

class KerasVisualLayer(BaseModel):
    type: KerasVisualLayerType = ...
    neurons: Optional[StrictInt] = Field(None, description="Number of neurons or filters in this layer (only for dense, conv1d, conv2d) or in the final conv2d layer (only for transfer layers)")
    kernel_size: Optional[StrictInt] = Field(None, alias="kernelSize", description="Kernel size for the convolutional layers (only for conv1d, conv2d)")
    dropout_rate: Optional[float] = Field(None, alias="dropoutRate", description="Fraction of input units to drop (only for dropout) or in the final layer dropout (only for transfer layers)")
    columns: Optional[StrictInt] = Field(None, description="Number of columns for the reshape operation (only for reshape)")
    stack: Optional[StrictInt] = Field(None, description="Number of convolutional layers before the pooling layer (only for conv1d, conv2d)")
    enabled: Optional[StrictBool] = None
    organization_model_id: Optional[StrictInt] = Field(None, alias="organizationModelId", description="Custom transfer learning model ID (when type is set to transfer_organization)")
    __properties = ["type", "neurons", "kernelSize", "dropoutRate", "columns", "stack", "enabled", "organizationModelId"]

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
    def from_json(cls, json_str: str) -> KerasVisualLayer:
        """Create an instance of KerasVisualLayer from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> KerasVisualLayer:
        """Create an instance of KerasVisualLayer from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return KerasVisualLayer.construct(**obj)

        _obj = KerasVisualLayer.construct(**{
            "type": obj.get("type"),
            "neurons": obj.get("neurons"),
            "kernel_size": obj.get("kernelSize"),
            "dropout_rate": obj.get("dropoutRate"),
            "columns": obj.get("columns"),
            "stack": obj.get("stack"),
            "enabled": obj.get("enabled"),
            "organization_model_id": obj.get("organizationModelId")
        })
        return _obj
