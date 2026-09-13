import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6 import QtCore, QtGui, QtTest, QtWidgets

    from anylabeling.views.labeling.shape import Shape
    from anylabeling.views.labeling.widgets.canvas import Canvas

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


@unittest.skipUnless(
    PYQT_AVAILABLE, "PyQt6 is required for canvas shape selection tests"
)
class TestCanvasShapeSelection(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.canvas = Canvas(parent=None)
        self.canvas.pixmap = QtGui.QPixmap(200, 200)
        self.canvas.pixmap.fill(QtGui.QColor("black"))
        self.canvas.resize(200, 200)
        self.canvas.selection_changed.connect(self._set_selection)

    def tearDown(self):
        self.canvas.close()
        self.app.processEvents()

    def _set_selection(self, shapes):
        for shape in self.canvas.selected_shapes:
            shape.selected = False
        self.canvas.selected_shapes = shapes
        for shape in shapes:
            shape.selected = True

    def _move_mouse(self, x, y):
        point = QtCore.QPointF(x, y)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            point,
            point,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.mouseMoveEvent(event)

    @staticmethod
    def make_rectangle(label, left, top, right, bottom):
        shape = Shape(label=label, shape_type="rectangle")
        shape.points = [
            QtCore.QPointF(left, top),
            QtCore.QPointF(right, top),
            QtCore.QPointF(right, bottom),
            QtCore.QPointF(left, bottom),
        ]
        shape.close()
        return shape

    def test_click_selects_nested_shape_below_larger_shape(self):
        inner = self.make_rectangle("inner", 40, 40, 80, 80)
        outer = self.make_rectangle("outer", 10, 10, 150, 150)
        self.canvas.shapes = [inner, outer]

        self.canvas.select_shape_point(
            QtCore.QPointF(60, 60), multiple_selection_mode=False
        )

        self.assertEqual(self.canvas.selected_shapes, [inner])

    def test_hover_finds_nested_shape_vertex_below_larger_shape(self):
        inner = self.make_rectangle("inner", 40, 40, 80, 80)
        outer = self.make_rectangle("outer", 10, 10, 150, 150)
        self.canvas.shapes = [inner, outer]

        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(40, 40),
            QtCore.QPointF(40, 40),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.mouseMoveEvent(event)

        self.assertIs(self.canvas.h_shape, inner)
        self.assertEqual(self.canvas.h_vertex, 0)

    def test_clicking_unselected_vertex_selects_its_shape(self):
        inner = self.make_rectangle("inner", 40, 40, 80, 80)
        outer = self.make_rectangle("outer", 10, 10, 150, 150)
        self.canvas.shapes = [inner, outer]
        self.canvas.h_shape = inner
        self.canvas.h_vertex = 0

        self.canvas.select_shape_point(
            QtCore.QPointF(40, 40), multiple_selection_mode=False
        )

        self.assertEqual(self.canvas.selected_shapes, [inner])

    def test_auto_highlight_click_allows_move_until_mouse_leaves(self):
        shape = self.make_rectangle("shape", 40, 40, 80, 80)
        self.canvas.shapes = [shape]
        self.canvas.h_shape_is_hovered = True

        self._move_mouse(60, 60)
        QtTest.QTest.mouseClick(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(60, 60),
        )

        self.assertEqual(self.canvas.selected_shapes, [shape])
        before = shape.points[0].x()
        with mock.patch.object(self.canvas, "update") as update:
            QtTest.QTest.keyClick(self.canvas, QtCore.Qt.Key.Key_Right)

        self.assertEqual(shape.points[0].x(), before + 1.0)
        update.assert_called_once_with()
        self.assertFalse(self.canvas.moving_shape)

        self._move_mouse(180, 180)

        self.assertEqual(self.canvas.selected_shapes, [])

    def test_finishing_shape_move_requests_repaint(self):
        shape = self.make_rectangle("shape", 40, 40, 80, 80)
        self.canvas.shapes = [shape]
        self.canvas.selected_shapes = [shape]
        self.canvas.moving_shape = True

        with mock.patch.object(self.canvas, "update") as update:
            self.canvas.store_moving_shape()

        update.assert_called_once_with()

    def test_vertex_proximity_outweighs_smaller_overlapping_area(self):
        outer = self.make_rectangle("outer", 10, 10, 150, 150)
        overlapping = self.make_rectangle("overlapping", -20, -20, 30, 30)
        self.canvas.shapes = [outer, overlapping]

        candidates = self.canvas._shape_hit_candidates(QtCore.QPointF(12, 12))

        self.assertEqual(candidates[:2], [outer, overlapping])

    def test_equal_area_overlap_keeps_top_shape_priority(self):
        lower = self.make_rectangle("lower", 20, 20, 100, 100)
        upper = self.make_rectangle("upper", 20, 20, 100, 100)
        self.canvas.shapes = [lower, upper]

        candidates = self.canvas._shape_hit_candidates(QtCore.QPointF(60, 60))

        self.assertEqual(candidates, [upper, lower])

    def test_hidden_nested_shape_is_not_a_candidate(self):
        inner = self.make_rectangle("inner", 40, 40, 80, 80)
        outer = self.make_rectangle("outer", 10, 10, 150, 150)
        self.canvas.shapes = [inner, outer]
        self.canvas.visible[inner] = False

        candidates = self.canvas._shape_hit_candidates(QtCore.QPointF(60, 60))

        self.assertEqual(candidates, [outer])

    @staticmethod
    def make_polygon(label, points):
        shape = Shape(label=label, shape_type="polygon")
        shape.points = [QtCore.QPointF(x, y) for x, y in points]
        shape.close()
        return shape

    def _drag_area(self, start, end, additive=False):
        modifiers = (
            QtCore.Qt.KeyboardModifier.ControlModifier
            if additive
            else QtCore.Qt.KeyboardModifier.NoModifier
        )
        QtTest.QTest.mousePress(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            modifiers,
            QtCore.QPoint(*start),
        )
        point = QtCore.QPointF(*end)
        self.canvas.mouseMoveEvent(
            QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                point,
                point,
                QtCore.Qt.MouseButton.NoButton,
                QtCore.Qt.MouseButton.LeftButton,
                modifiers,
            )
        )
        QtTest.QTest.mouseRelease(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            modifiers,
            QtCore.QPoint(*end),
        )

    def test_area_drag_selects_intersecting_polygons_without_moving(self):
        inside = self.make_polygon("inside", [(30, 30), (60, 30), (45, 60)])
        crossing = self.make_polygon(
            "crossing", [(80, 80), (150, 80), (150, 150)]
        )
        outside = self.make_polygon(
            "outside", [(140, 140), (180, 140), (180, 180)]
        )
        hidden = self.make_polygon("hidden", [(40, 40), (70, 40), (55, 70)])
        self.canvas.shapes = [inside, crossing, outside, hidden]
        self.canvas.visible[hidden] = False
        before = [list(shape.points) for shape in self.canvas.shapes]
        with mock.patch.object(self.canvas, "scroll_request") as scroll:
            self._drag_area((10, 10), (100, 100))

        self.assertEqual(self.canvas.selected_shapes, [inside, crossing])
        self.assertEqual([s.points for s in self.canvas.shapes], before)
        self.assertFalse(self.canvas.moving_shape)
        scroll.emit.assert_not_called()

    def test_area_drag_uses_polygon_geometry_instead_of_bounding_box(self):
        triangle = self.make_polygon(
            "triangle", [(20, 20), (160, 20), (20, 160)]
        )
        self.canvas.shapes = [triangle]
        self._drag_area((170, 170), (120, 120))
        self.assertEqual(self.canvas.selected_shapes, [])

    def test_reverse_area_drag_adds_to_existing_selection(self):
        first = self.make_polygon("first", [(30, 30), (60, 30), (45, 60)])
        second = self.make_polygon(
            "second", [(110, 110), (140, 110), (125, 140)]
        )
        self.canvas.shapes = [first, second]
        self._set_selection([first])
        self._drag_area((160, 160), (90, 90), additive=True)
        self.assertEqual(self.canvas.selected_shapes, [first, second])

    def test_area_drag_replaces_selection_and_survives_auto_highlight(self):
        first = self.make_polygon("first", [(30, 30), (60, 30), (45, 60)])
        second = self.make_polygon(
            "second", [(110, 110), (140, 110), (125, 140)]
        )
        self.canvas.shapes = [first, second]
        self.canvas.h_shape_is_hovered = True
        self._set_selection([second])
        self._drag_area((10, 10), (80, 80))
        self._move_mouse(190, 190)
        self._move_mouse(125, 120)
        self.assertEqual(self.canvas.selected_shapes, [first])

    def test_click_on_empty_space_clears_selection_unless_ctrl_pressed(self):
        shape = self.make_polygon("shape", [(30, 30), (60, 30), (45, 60)])
        self.canvas.shapes = [shape]
        self._set_selection([shape])
        self._drag_area((180, 180), (181, 181), additive=True)
        self.assertEqual(self.canvas.selected_shapes, [shape])
        self._drag_area((180, 180), (181, 181))
        self.assertEqual(self.canvas.selected_shapes, [])

    def test_area_drag_accounts_for_zoom_and_center_offset(self):
        shape = self.make_polygon("shape", [(30, 30), (60, 30), (45, 60)])
        self.canvas.shapes = [shape]
        self.canvas.scale = 2.0
        self.canvas.resize(600, 600)
        offset = self.canvas.offset_to_center()
        start = (QtCore.QPointF(10, 10) + offset) * self.canvas.scale
        end = (QtCore.QPointF(80, 80) + offset) * self.canvas.scale
        self._drag_area(
            (int(start.x()), int(start.y())), (int(end.x()), int(end.y()))
        )
        self.assertEqual(self.canvas.selected_shapes, [shape])

    def test_escape_cancels_area_selection(self):
        shape = self.make_polygon("shape", [(30, 30), (60, 30), (45, 60)])
        self.canvas.shapes = [shape]
        QtTest.QTest.mousePress(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(10, 10),
        )
        QtTest.QTest.keyClick(self.canvas, QtCore.Qt.Key.Key_Escape)
        QtTest.QTest.mouseRelease(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(80, 80),
        )
        self.assertEqual(self.canvas.selected_shapes, [])
        self.assertIsNone(self.canvas._selection_drag_start)

    def test_space_drag_still_pans(self):
        QtTest.QTest.keyPress(self.canvas, QtCore.Qt.Key.Key_Space)
        with mock.patch.object(self.canvas, "scroll_request") as scroll:
            self._drag_area((10, 10), (80, 80))
        QtTest.QTest.keyRelease(self.canvas, QtCore.Qt.Key.Key_Space)
        self.assertTrue(scroll.emit.called)
        self.assertEqual(self.canvas.selected_shapes, [])

    def test_area_selection_can_move_left_by_keyboard_and_respects_locks(self):
        movable = self.make_polygon("movable", [(30, 30), (60, 30), (45, 60)])
        locked = self.make_polygon("locked", [(70, 30), (90, 30), (80, 60)])
        locked.locked = True
        self.canvas.shapes = [movable, locked]
        self._drag_area((10, 10), (110, 80))
        self.assertEqual(self.canvas.selected_shapes, [movable, locked])
        QtTest.QTest.keyClick(self.canvas, QtCore.Qt.Key.Key_Left)
        self.assertEqual(movable.points[0], QtCore.QPointF(29, 30))
        self.assertEqual(locked.points[0], QtCore.QPointF(70, 30))

    def test_area_selection_rectangle_is_painted_and_cleared_on_release(self):
        self.canvas.cross_line_show = False
        self.canvas.show()
        self.app.processEvents()
        self._move_mouse(100, 100)
        QtTest.QTest.mousePress(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(10, 10),
        )
        self._move_mouse(100, 100)
        preview = self.canvas.grab().toImage()
        self.assertNotEqual(preview.pixelColor(50, 50), QtGui.QColor("black"))
        QtTest.QTest.mouseRelease(
            self.canvas,
            QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(100, 100),
        )
        finished = self.canvas.grab().toImage()
        self.assertEqual(finished.pixelColor(50, 50), QtGui.QColor("black"))
