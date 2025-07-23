from trainer.common.models.bifpn.bifpn_3d import BiFPN


def test_bifpn_num_layers():
    """Test different number of layers"""
    BiFPN([55, 55, 32, 30, 28], pyramid_channels=55, num_layers=2)
    BiFPN([1, 2, 3, 4], pyramid_channels=55, num_layers=2)
