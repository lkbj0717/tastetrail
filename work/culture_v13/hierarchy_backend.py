"""Explicit version routing; importing a backend never executes model calls."""
def get_backend(version=None):
    import hierarchical_agent
    if version is None or version==hierarchical_agent.VERSION:return hierarchical_agent
    if version=='hierarchical_v2_repair':
        import hierarchical_repair
        return hierarchical_repair
    raise ValueError('Unsupported hierarchical version: '+str(version))
