#!/usr/bin/env python3
"""Simple Python demo for shape_based_matching bindings."""

from pathlib import Path

import shape_based_matching_py as sbm


def main() -> None:
    root = Path(__file__).resolve().parent

    detector = sbm.Detector(num_features=128, T=[4, 8], weak_thresh=30.0, strong_thresh=60.0)

    # 1) train one template from case0
    template_img = root / "test/case0/templ/circle.png"
    template_id = detector.add_template(str(template_img), class_id="circle")
    if template_id < 0:
        raise RuntimeError("add_template failed, please lower num_features or change image")

    # 2) optional: save templates for later loading
    detector.write_classes(str(root / "test/case0/%s_templ.yaml"))

    # 3) match on a test image
    test_img = root / "test/case0/1.jpg"
    matches = detector.match(str(test_img), threshold=90.0, class_ids=["circle"])

    print(f"template_id={template_id}, total_matches={len(matches)}")
    for i, m in enumerate(matches[:5]):
        info = detector.get_template_info(m["class_id"], m["template_id"], pyramid_level=0)
        center_x = m["x"] + info["width"] // 2
        center_y = m["y"] + info["height"] // 2
        print(
            f"[{i}] class={m['class_id']} score={m['similarity']:.2f} "
            f"center=({center_x},{center_y}) size=({info['width']}x{info['height']})"
        )


if __name__ == "__main__":
    main()
