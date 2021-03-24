"""
Some utility functions
"""
from typing import Dict, List, Tuple, Optional, Union


def dict_get_compat(in_dict: Dict[str, any], current_key: Optional[str], compat_keys: List[str], default: any = None) -> Tuple[any, Optional[str]]:
    """
    Given a dictionary, an expected key, and a list of older compatibility keys, return the first found value
    from the dict and which key it was.
    """
    test_keys = [current_key] + compat_keys if current_key else compat_keys
    for key in test_keys:
        if key in in_dict:
            return in_dict[key], key
    return default, None


def decode_dict_bytes_as_str(in_dict: Dict[any, any], encoding="utf-8"):
    """
    Recursively decode any byte values in the dict as strings with .decode()
    """
    # modifies the dict in place
    for key, val in in_dict.items():
        if isinstance(val, bytes):
            in_dict[key] = val.decode(encoding)
        elif isinstance(val, list) or isinstance(val, tuple):
            in_dict[key] = decode_list_bytes_as_str(val, encoding=encoding)
        elif isinstance(val, dict):
            decode_dict_bytes_as_str(val)


def decode_list_bytes_as_str(in_list: Union[List[any], Tuple[any]], encoding="utf-8"):
    """
    Recursively decodes any byte values in the list as strings with .decode()
    """
    # make the return the same type as this one
    decoded_list = []
    for item in in_list:
        if isinstance(item, bytes):
            decoded_list.append(item.decode(encoding))
        elif isinstance(item, list) or isinstance(item, tuple):
            decoded_list.append(decode_list_bytes_as_str(item))
        else:
            decoded_list.append(item)

    return type(in_list)(decoded_list)


def list_or_tuple(item):
    """
    returns true if the item is a list or tuple
    """
    return isinstance(item, list) or isinstance(item, tuple)
