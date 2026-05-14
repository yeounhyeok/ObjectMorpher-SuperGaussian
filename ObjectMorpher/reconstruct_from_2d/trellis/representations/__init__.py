from .radiance_field import Strivec
from .octree import DfsOctree as Octree
from .gaussian import Gaussian
try:
    from .mesh import MeshExtractResult
except ImportError:
    MeshExtractResult = None  # kaolin/nvdiffrast unavailable; gaussian-only lifting still works
