"Introduced from code_arctic"
import torch
import numpy as np
from pytorch3d.ops import knn_points, knn_gather
import trimesh
from libzhifan.geometry import SimpleMesh
from libzhifan.geometry import visualize as geo_vis
import torch.nn.functional as F

# from nnutils.mesh_utils_extra import compute_vert_normals
def compute_face_angles(verts: torch.Tensor,
                        faces: torch.Tensor) -> torch.Tensor:
    """ compute face areas 

    Args:
        verts: (B, V, 3) or (V, 3)
        faces: (F, 3) LongTensor
    Returns:
        face_angles: (B, F, 3) or (F, 3)
    """
    ndim_old = verts.ndim
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    bs = verts.size(0)
    nf = faces.size(0)
    vfaces = verts[:, faces]  # (B, F, 3, 3)

    v1, v2, v3 = torch.split(vfaces, 1, dim=2)  # each (B, F, 1, 3)
    e12 = v2 - v1
    e23 = v3 - v2
    e31 = v1 - v3
    l12 = e12.norm(p=2, dim=-1)
    l23 = e23.norm(p=2, dim=-1)
    l31 = e31.norm(p=2, dim=-1)
    va1 = torch.sum(e12 * -e31, dim=-1) / (l12 * l31)
    va2 = torch.sum(e23 * -e12, dim=-1) / (l23 * l12)
    va3 = torch.sum(e31 * -e23, dim=-1) / (l31 * l23)
    face_cosines = torch.cat([va1, va2, va3], dim=2).view(bs, nf, 3)
    face_angles = face_cosines.acos_()
    if ndim_old == 2:
        face_angles = face_angles.squeeze_(0)
    return face_angles

def compute_face_normals(verts: torch.Tensor,
                         faces: torch.Tensor) -> torch.Tensor:
    """ compute face normals 
        (it doesn't matter which vertex to use as they are co-planar)
            fn[i] = e12 x e13

    Args:
        verts: (B, V, 3) or (V, 3)
        faces: (F, 3) LongTensor
    Returns:
        face_normals: (B, F, 3) or (F, 3)
    """
    ndim_old = verts.ndim
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    vfaces = verts[:, faces]  # (B, F, 3, 3)

    v1, v2, v3 = torch.split(vfaces, 1, dim=2)  # each (B, F, 1, 3)
    e12 = v2 - v1
    e31 = v1 - v3
    vn1 = torch.cross(e12, -e31, dim=-1)  # (B, F, 1, 3)
    vn1 = F.normalize(vn1, p=2, dim=-1)
    vn1 = vn1.squeeze_(2)
    if ndim_old == 2:
        vn1 = vn1.squeeze_(0)
    return vn1

def compute_vert_normals(verts: torch.Tensor,
                         faces: torch.Tensor,
                         method: str = 'f') -> torch.Tensor:
    """
    Args:
        verts: (B, V, 3) or (V, 3)
        faces: (F, 3) LongTensor
        method: one of {'v', 'f'}
            if method == 'v', compute as 
                normed mean of all connected VERTICES,
                i.e. vn[i] = mean( eij x eik )

            elif method == 'f', compte as
                meaned weighted mean of all connected FACES, as Trimesh,
                where weights are computed according to angles,
                i.e. more flat faces contribute larger

    Returns:
        vert_normals: (B, V, 3) or (V, 3)
    """
    ndim_old = verts.ndim
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    bs = verts.size(0)
    nf = faces.size(0)

    if method == 'v':
        vfaces = verts[:, faces]  # (B, F, 3, 3)
        v1, v2, v3 = torch.split(vfaces, 1, dim=2)  # each (B, F, 1, 3)
        e12 = v2 - v1
        e23 = v3 - v2
        e31 = v1 - v3
        vn1 = torch.cross(e12, -e31, dim=-1)  # e12 x e13
        vn2 = torch.cross(e23, -e12, dim=-1)
        vn3 = torch.cross(e31, -e23, dim=-1)
        src = torch.cat([vn1, vn2, vn3], dim=2).view(bs, nf*3, 3)
        src = F.normalize(src, p=2, dim=2)

        index = faces.view(1, nf*3, 1).expand(bs, nf*3, 3)
        vn = torch.zeros_like(verts).scatter_add_(dim=1, index=index, src=src)
    elif method == 'f':
        face_normals = compute_face_normals(verts, faces).unsqueeze_(2)  # (B, F, 1, 3)
        face_angles = compute_face_angles(verts, faces).unsqueeze_(3)  # (B, F, 3, 1)
        src = face_normals * face_angles  # (B, F, 3, 3)
        src = src.view(bs, nf*3, 3)
        index = faces.view(1, nf*3, 1).expand(bs, nf*3, 3)
        vn = torch.zeros_like(verts).scatter_add_(dim=1, index=index, src=src)
    else:
        raise ValueError(f"method {method} not understood.")

    vn = F.normalize(vn, p=2, dim=2)
    if ndim_old == 2:
        vn = vn.squeeze_(0)
    return vn


def pcd_mesh_distance_approx(verts, faces, verts_query, 
                             along_normal=True, debug=False):
    """
    Artgs:
        verts1: (N, V1, 3)
        faces1: (F1, 3)
        verts2: (N, V2, 3)
    
    Returns:
        dist2: (N, V2)
    """
    assert isinstance(verts, torch.Tensor)
    assert isinstance(faces, torch.Tensor)
    vn = compute_vert_normals(verts, faces)
    _dists, idx, nn = knn_points(verts_query, verts, K=1, return_nn=True)
    vn_ = knn_gather(vn, idx)  # (N, V2, 3)
    vn_ = vn_[:, :, 0, :]
    nn = nn[:, :, 0, :]

    disp = verts_query - nn
    disp_norm = disp.norm(dim=2)  # (N, V2)
    disp_dir = disp / disp_norm.unsqueeze(-1)  # (N, V2, 3)
    if along_normal:
        cos = (disp_dir * vn_).sum(dim=2)  # (N, V2)
        signed_dist = disp_norm * cos
    else:
        cos = (disp_dir * vn_).sum(dim=2)  # (N, V2)
        signed_dist = torch.copysign(disp_norm, cos)
    if debug:
        i_query = signed_dist[0].argmax()
        i2 = idx[0, i_query]
        print(i_query, i2)
    return signed_dist


def mesh_distance_approx(verts1: torch.Tensor, 
                         verts2: torch.Tensor,
                         faces1: torch.Tensor,
                         faces2: torch.Tensor,
                         along_normal=True):
    """
    Use vertices to approximate mesh distance
    Args:
        verts1: (N, V1, 3)
        verts2: (N, V2, 3)
        faces1: (F1, 3)
        faces2: (F2, 3)
        along_normal: if True, measure distance along the normal direction.
    
    Returns: 
        dist1: (N, V1)
        dist2: (N, V2)
    """
    d1 = pcd_mesh_distance_approx(verts2, faces2, verts1, along_normal=along_normal)
    d2 = pcd_mesh_distance_approx(verts1, faces1, verts2, along_normal=along_normal)
    return d1, d2


def single_pcd_mesh_distance_approx(verts, faces, verts_query, debug=False):
    """
    Artgs:
        verts1: (V1, 3)
        faces1: (F1, 3)
        verts2: (V2, 3)
    
    Returns:
        dist2: (V2)
    """
    d = pcd_mesh_distance_approx(verts[None], faces, verts_query[None], debug=True)
    return d[0]


def single_mesh_distance_approx(verts1: torch.Tensor, 
                                verts2: torch.Tensor,
                                faces1: torch.Tensor,
                                faces2: torch.Tensor):
    """
    Use vertices to approximate mesh distance
    Args:
        verts1: (V1, 3)
        verts2: (V2, 3)
        faces1/2: (F1/F2, 3)
    
    Returns:
        dist1: (V1)
        dist2: (V2)
    """
    d1, d2 = mesh_distance_approx(
        verts1[None], verts2[None], faces1, faces2)
    d1 = d1.squeeze()
    d2 = d2.squeeze()
    return d1, d2


def mesh_distance_visualize(verts1, verts2, faces1, faces2, up_thr=0.001):
    """
    Args:
        verts1/verts2: (V1, 3)
        faces1/faces2: (F, 3)
    """
    d1, d2 = single_mesh_distance_approx(verts1, verts2, faces1, faces2)
    d1 = d1.detach().cpu().numpy()
    d2 = d2.detach().cpu().numpy()
    m1 = SimpleMesh(verts1, faces1)
    m2 = SimpleMesh(verts2, faces2)
    
    # up_thr: (-inf, 0.1cm) as in-contact
    m1 = geo_vis.color_verts(m1, d1 < up_thr, (255, 0, 0))
    m2 = geo_vis.color_verts(m2, d2 < up_thr, (255, 0, 0))
    return m1, m2


def mesh_connection_visualize(verts1,
                              verts2,
                              faces1,
                              faces2,
                              i1,
                              i2):
    """
    Given vertex index i1 in mesh1, vertex index i2 in mesh2,
    return mesh1 colored at p1, mesh2 colored at p2, and
        a link path connecting p1->p2
    
    Args:
        verts: (V, 3)
        faces: (F, 3)
    
    Returns:
        mesh1, mesh2, path
    """
    m1 = SimpleMesh(verts1, faces1)
    m2 = SimpleMesh(verts2, faces2)
    m1 = geo_vis.color_verts(m1, [i1], (255, 0, 0))
    m2 = geo_vis.color_verts(m2, [i2], (0, 0, 255))
    p1 = m1.vertices[i1]  
    p2 = m2.vertices[i2]
    path = geo_vis.create_path_cone(p1, p2)
    return m1, m2, path