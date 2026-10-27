"""finetune/labeling/label_window.py: the window plumbing that can be tested without a window."""
from unittest.mock import patch

from finetune.labeling import label_window


# --------------------------------------------------------------------------- the mouse's side buttons
def fake_user32(held, in_front=True):
    """A user32 whose side buttons are the ones in `held` (a set the test edits) and whose window is
    the tool's when `in_front`."""
    from types import SimpleNamespace
    return SimpleNamespace(
        GetAsyncKeyState=lambda vk: label_window.KEY_DOWN if vk in held else 0,
        FindWindowW=lambda cls, title: 7,
        GetForegroundWindow=lambda: 7 if in_front else 9)


def test_side_buttons_are_the_arrows_once_per_press_and_only_with_the_window_in_front():
    held = set()
    buttons = label_window.SideButtons()
    with patch.object(label_window, "_user32", fake_user32(held)):
        assert buttons.key() == -1
        held.add(label_window.VK_XBUTTON2)                      # forward: the right arrow, once
        assert buttons.key() == label_window.VK_RIGHT
        assert buttons.key() == -1                              # still held: not a second press
        held.clear()
        held.add(label_window.VK_XBUTTON1)                      # back: the left arrow
        assert buttons.key() == label_window.VK_LEFT
        held.clear()
        assert buttons.key() == -1
    with patch.object(label_window, "_user32", fake_user32(held, in_front=False)):
        held.add(label_window.VK_XBUTTON2)
        assert buttons.key() == -1                              # a browser's forward, not ours
    with patch.object(label_window, "_user32", None):
        assert buttons.key() == -1
