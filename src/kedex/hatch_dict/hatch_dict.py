from flatten_dict import flatten
from kedro.utils import load_obj
from six import iteritems
import operator
from typing import Union, List, Iterable  # NOQA
from types import MethodType


class HatchDict:
    def __init__(
        self,
        egg,  # type: Union[dict, List]
        lookup={},  # type: dict
        support_nested_keys=True,  # type: bool
        self_lookup_key="=",  # type: str
        support_import=True,  # type: bool
        additional_import_modules=["kedex"],  # type: Union[List, str]
        obj_key="=",  # type: str
        eval_parentheses=True,  # type: bool
    ):
        # type: (...) -> None

        assert egg.__class__.__name__ in {"dict", "list"}
        assert lookup.__class__.__name__ in {"dict"}
        assert support_nested_keys.__class__.__name__ in {"bool"}
        assert self_lookup_key.__class__.__name__ in {"str"}
        assert additional_import_modules.__class__.__name__ in {"list", "str"}
        assert obj_key.__class__.__name__ in {"str"}

        aug_egg = {}
        if isinstance(egg, dict):
            if support_nested_keys:
                aug_egg = dot_flatten(egg)
            aug_egg.update(egg)
        self.aug_egg = aug_egg

        self.egg = egg

        self.lookup = {}

        self.lookup.update(_builtin_funcs())
        self.lookup.update(lookup)

        self.self_lookup_key = self_lookup_key
        self.support_import = support_import
        self.additional_import_modules = (
            [additional_import_modules]
            if isinstance(additional_import_modules, str)
            else additional_import_modules or [__name__]
        )
        self.obj_key = obj_key
        self.eval_parentheses = eval_parentheses

        self.warmed_egg = None
        self.snapshot = None

    def get(
        self,
        key=None,  # type: Union[str, int]
        default=None,  # type: Any
        lookup={},  # type: dict
    ):
        # type: (...) -> Any

        assert key.__class__.__name__ in {"str", "int"}
        assert lookup.__class__.__name__ in {"dict"}

        if key is None:
            d = self.egg
        else:
            if isinstance(self.egg, dict):
                d = self.aug_egg.get(key, default)
            if isinstance(self.egg, list):
                assert isinstance(key, int)
                d = self.egg[key] if (0 <= key < len(self.egg)) else default

        if self.self_lookup_key:
            s = dict()
            while d != s:
                d, s = _dfs_apply(
                    d_input=d,
                    hatch_args=dict(lookup=self.aug_egg, obj_key=self.self_lookup_key),
                )
            self.warmed_egg = d

        if self.eval_parentheses:
            d, s = _dfs_apply(
                d_input=d, hatch_args=dict(eval_parentheses=self.eval_parentheses)
            )
            self.warmed_egg = d

        lookup_input = {}
        lookup_input.update(self.lookup)
        lookup_input.update(lookup)

        if isinstance(self.egg, dict):
            forcing_module = self.egg.get("FORCING_MODULE", "")

        for m in self.additional_import_modules:
            d, s = _dfs_apply(
                d_input=d,
                hatch_args=dict(
                    lookup=lookup_input,
                    support_import=self.support_import,
                    default_module=m,
                    forcing_module=forcing_module,
                    obj_key=self.obj_key,
                ),
            )
        self.snapshot = s
        return d

    def get_params(self):
        return self.snapshot


def _dfs_apply(
    d_input,  # type: Any
    hatch_args,  # type: dict
):
    # type: (...) -> Any

    eval_parentheses = hatch_args.get("eval_parentheses", False)  # type: bool
    lookup = hatch_args.get("lookup", dict())  # type: dict
    support_import = hatch_args.get("support_import", False)  # type: bool
    default_module = hatch_args.get("default_module", "")  # type: str
    forcing_module = hatch_args.get("forcing_module", "")  # type: str
    obj_key = hatch_args.get("obj_key", "=")  # type: str

    d = d_input
    s = d_input

    if isinstance(d_input, dict):

        obj_str = d_input.get(obj_key)

        d, s = {}, {}
        for k, v in iteritems(d_input):
            d[k], s[k] = _dfs_apply(v, hatch_args)

        if obj_str:
            if obj_str in lookup:
                a = lookup.get(obj_str)
                d = _hatch(d, a, obj_key=obj_key)
            elif support_import:
                if forcing_module:
                    obj_str = "{}.{}".format(forcing_module, obj_str.rsplit(".", 1)[-1])
                a = load_obj(obj_str, default_obj_path=default_module)
                d = _hatch(d, a, obj_key=obj_key)

    if isinstance(d_input, list):

        d, s = [], []
        for v in d_input:
            _d, _s = _dfs_apply(v, hatch_args)
            d.append(_d)
            s.append(_s)

    if isinstance(d_input, str):
        if (
            eval_parentheses
            and len(d_input) >= 2
            and d_input[0] == "("
            and d_input[-1] == ")"
        ):
            d = eval(d)

    return d, s


def _hatch(
    d,  # type: dict
    a,  # type: Any
    obj_key="=",  # type: str
    pos_arg_key="_",  # type: str
    attr_key=".",  # type: str
):
    d.pop(obj_key)
    if d:
        assert callable(a)

        pos_args = d.pop(pos_arg_key, None)
        if pos_args is None:
            pos_args = []
        if not isinstance(pos_args, list):
            pos_args = [pos_args]

        attribute_name = d.pop(attr_key, None)
        for k in d:
            assert isinstance(
                k, str
            ), "Non-string key '{}' in '{}' is not valid for callable: '{}'.".format(
                k, d, a.__name__
            )
        d = a(*pos_args, **d)
        if attribute_name:
            d = getattr(d, attribute_name)
            # if isinstance(d, MethodType):
            #     d = lambda *args: d(args[0])
    else:
        d = a
    return d


def dot_flatten(d):
    def dot_reducer(k1, k2):
        return k1 + "." + k2 if k1 else k2

    return flatten(d, reducer=dot_reducer)


def pass_(*argsignore, **kwargsignore):
    return None


def pass_through(*args, **kwargs):
    return args[0] if args else list(kwargs.values())[0] if kwargs else None


def pass_func(arg):
    def _pass_func(*argsignore, **kwargsignore):
        return arg

    return _pass_func


def _builtin_funcs():
    return dict(pass_=pass_, pass_through=pass_through, pass_func=pass_func)
