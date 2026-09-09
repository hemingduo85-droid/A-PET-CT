import site
import sys


def _drop_user_site_packages():
    user_site = site.getusersitepackages()
    user_base = site.getuserbase()
    blocked = {user_site, user_base}
    sys.path[:] = [
        path for path in sys.path
        if path and not any(path.startswith(prefix) for prefix in blocked)
    ]
    site.ENABLE_USER_SITE = False


_drop_user_site_packages()
