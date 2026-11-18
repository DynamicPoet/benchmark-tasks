import numpy as np
from functional_pads import FunctionalPads


def test_observed_off_flat_plane_contact_is_on_the_real_finite_inner_surface():
    pads=FunctionalPads()
    # Native loaded contact, reconstructed in the actual tip frame by stock FK.
    # Its TCP/q flat-plane residual was -0.661mm; the allowance stays0.500mm.
    contact=np.array([.035552456464,-.045172677809,-.078146355333])
    assert pads.contains_surface('tip_left',contact)
    # The face's mathematical extension is not the finite functional pad.
    assert not pads.contains_surface('tip_left',contact+[.060,0.,0.])
    assert not pads.contains_surface('tip_left',contact+[0.,.004,0.])
