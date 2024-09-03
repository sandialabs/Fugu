def is_metadata_list_of_dicts(metadata):
    if isinstance(metadata, list):
        return True
    elif isinstance(metadata, dict):
        return False
    else:
        raise ValueError("Metadata is not a dictionary or a list of dictionaries.")

def is_metadata_key_present(metadata, metadata_key):
    if metadata_key in metadata:
        return True
    else:
        return False 

def get_metadata_key_value(metadata, metadata_key):
    # isMetadataAList = is_metadata_list_of_dicts(metadata)
    # dictionary = metadata[0] if isMetadataAList else metadata

    isKeyPresent = is_metadata_key_present(metadata, metadata_key)
    assert isKeyPresent
    return metadata[metadata_key]
