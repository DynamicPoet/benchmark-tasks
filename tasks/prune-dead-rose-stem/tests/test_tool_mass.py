"""Physical catalogue mass, finite solid moments and a realizable internal spine."""
import numpy as np
import pytest
from tool_mass import moments
from usdkit import box
import scene_assets as s


def test_finite_box_mass_tensor_and_rigid_translation():
    size=np.array([.011,.030,.003]); center=np.array([.007,-.024,.001])
    mesh=box(size, center=center)
    v,p,q=moments(mesh.v,mesh.f)
    assert v==pytest.approx(np.prod(size),rel=1e-12)
    assert p/v==pytest.approx(center,abs=1e-12)
    assert q/v-np.outer(center,center)==pytest.approx(np.diag(size**2/12),abs=1e-15)


def test_reference_class_mass_from_components_and_full_inertias():
    properties=s.secateurs_mass_properties()
    assert properties['total_mass_kg']==pytest.approx(.160,abs=1e-12)
    assert .0005 < properties['estimated_spine_thickness_m'] < .004
    for body in properties['bodies'].values():
        I=np.asarray(body['inertia_kg_m2']);R=np.asarray(body['principal_matrix'])
        assert R@np.diag(body['diagonal_inertia_kg_m2'])@R.T==pytest.approx(I,abs=1e-13)
        assert np.linalg.det(R)==pytest.approx(1.)
    com=sum(b['mass_kg']*np.array(b['com_m']) for b in properties['bodies'].values())/.16
    # The rear mass is represented physically; this is not the old hand-set COM.
    assert -.040 < com[1] < -.030
    assert .16*9.81*abs(-.080-com[1]) < .080


def test_spine_is_inside_the_finite_grip_at_both_ends():
    from scipy.spatial import ConvexHull
    from tool_mass import SPINE_FRONT,SPINE_BACK,SPINE_WIDTH
    thickness=s.secateurs_mass_properties()['estimated_spine_thickness_m']
    for side in(-1,1):
        grip=next(m for n,m,_ in s._closed_half(side)[0] if n=='grip')
        h=ConvexHull(grip.v)
        spine=box((SPINE_WIDTH,SPINE_FRONT-SPINE_BACK,thickness),
                  center=(-side*s.HANDLE_X,(SPINE_FRONT+SPINE_BACK)/2,0))
        assert np.max(spine.v@h.equations[:,:3].T+h.equations[:,3])<=1e-12


def test_pivot_colliders_preserve_each_actual_visible_solid_without_a_filled_rim():
    from scipy.spatial import ConvexHull
    parts=s._closed_half(-1)[0]
    pivots=s.pivot_collision_sections(parts)
    assert [n for n,_ in pivots]==['shoulder','bolt_head','nut']
    source=[next(m for n,m,_ in parts if n==name)for name in('boss','bolt')]
    # Exact closed-solid integration and the actual source vertices match.
    expected_volume=sum(moments(m.v,m.f)[0]for m in source)
    assert sum(moments(m.v,m.f)[0]for _,m in pivots)==pytest.approx(expected_volume,rel=1e-12)
    def ordered(points):return np.array(sorted(map(tuple,points)))
    assert np.array_equal(ordered(np.vstack([m.v for _,m in pivots])),ordered(np.vstack([m.v for m in source])))
    for _,m in pivots:
        assert ConvexHull(m.v).volume==pytest.approx(moments(m.v,m.f)[0],rel=1e-12)
    assert pivots[0][1].v[:,2].min()==pytest.approx(-.0058)
    assert pivots[0][1].v[:,2].max()==pytest.approx(.0034)
    text=open(s.write_secateurs()).read()
    assert 'h_boss_c'not in text
    for name,_ in pivots:assert f'h_{name}_c'in text
