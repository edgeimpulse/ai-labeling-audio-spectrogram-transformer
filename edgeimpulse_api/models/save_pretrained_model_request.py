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



from pydantic import BaseModel
from edgeimpulse_api.models.deploy_pretrained_model_request_model_info_input import DeployPretrainedModelRequestModelInfoInput
from edgeimpulse_api.models.deploy_pretrained_model_request_model_info_model import DeployPretrainedModelRequestModelInfoModel

class SavePretrainedModelRequest(BaseModel):
    input: DeployPretrainedModelRequestModelInfoInput = ...
    model: DeployPretrainedModelRequestModelInfoModel = ...
    __properties = ["input", "model"]

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
    def from_json(cls, json_str: str) -> SavePretrainedModelRequest:
        """Create an instance of SavePretrainedModelRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of input
        if self.input:
            _dict['input'] = self.input.to_dict()
        # override the default output from pydantic by calling `to_dict()` of model
        if self.model:
            _dict['model'] = self.model.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SavePretrainedModelRequest:
        """Create an instance of SavePretrainedModelRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return SavePretrainedModelRequest.construct(**obj)

        _obj = SavePretrainedModelRequest.construct(**{
            "input": DeployPretrainedModelRequestModelInfoInput.from_dict(obj.get("input")) if obj.get("input") is not None else None,
            "model": DeployPretrainedModelRequestModelInfoModel.from_dict(obj.get("model")) if obj.get("model") is not None else None
        })
        return _obj

