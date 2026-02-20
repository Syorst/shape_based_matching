#include "line2Dup.h"

#include <opencv2/imgcodecs.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;

namespace {

cv::Mat loadColor(const std::string &image_path)
{
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    return img;
}

cv::Mat loadMask(const std::string &mask_path)
{
    if (mask_path.empty())
    {
        return cv::Mat();
    }

    cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    if (mask.empty())
    {
        throw std::runtime_error("Failed to load mask: " + mask_path);
    }
    return mask;
}

class PyDetector
{
public:
    PyDetector(int num_features,
               const std::vector<int> &T,
               float weak_thresh,
               float strong_thresh)
        : detector_(num_features, T, weak_thresh, strong_thresh)
    {
    }

    int add_template(const std::string &image_path,
                     const std::string &class_id,
                     const std::string &mask_path = "",
                     int num_features = 0)
    {
        cv::Mat src = loadColor(image_path);
        cv::Mat mask = loadMask(mask_path);
        return detector_.addTemplate(src, class_id, mask, num_features);
    }

    int add_rotated_template(const std::string &class_id,
                             int zero_id,
                             float theta,
                             float center_x,
                             float center_y)
    {
        return detector_.addTemplate_rotate(class_id, zero_id, theta, cv::Point2f(center_x, center_y));
    }

    std::vector<py::dict> match(const std::string &image_path,
                                float threshold,
                                const std::vector<std::string> &class_ids = {},
                                const std::string &mask_path = "") const
    {
        cv::Mat src = loadColor(image_path);
        cv::Mat mask = loadMask(mask_path);
        auto matches = detector_.match(src, threshold, class_ids, mask);

        std::vector<py::dict> out;
        out.reserve(matches.size());
        for (const auto &m : matches)
        {
            py::dict d;
            d["x"] = m.x;
            d["y"] = m.y;
            d["similarity"] = m.similarity;
            d["class_id"] = m.class_id;
            d["template_id"] = m.template_id;
            out.emplace_back(std::move(d));
        }
        return out;
    }

    py::dict get_template_info(const std::string &class_id, int template_id, int pyramid_level = 0) const
    {
        const auto &templ = detector_.getTemplates(class_id, template_id);
        if (templ.empty())
        {
            throw std::runtime_error("Template pyramid is empty.");
        }
        if (pyramid_level < 0 || pyramid_level >= static_cast<int>(templ.size()))
        {
            throw std::runtime_error("Invalid pyramid level.");
        }

        const auto &t = templ[pyramid_level];
        py::dict d;
        d["width"] = t.width;
        d["height"] = t.height;
        d["tl_x"] = t.tl_x;
        d["tl_y"] = t.tl_y;
        d["pyramid_level"] = t.pyramid_level;
        return d;
    }

    void read_classes(const std::vector<std::string> &class_ids,
                      const std::string &format = "templates_%s.yml.gz")
    {
        detector_.readClasses(class_ids, format);
    }

    void write_classes(const std::string &format = "templates_%s.yml.gz") const
    {
        detector_.writeClasses(format);
    }

    int num_templates() const
    {
        return detector_.numTemplates();
    }

private:
    line2Dup::Detector detector_;
};

} // namespace

PYBIND11_MODULE(shape_based_matching_py, m)
{
    m.doc() = "Python bindings for shape_based_matching";

    py::class_<PyDetector>(m, "Detector")
        .def(py::init<int, const std::vector<int> &, float, float>(),
             py::arg("num_features") = 128,
             py::arg("T") = std::vector<int>{4, 8},
             py::arg("weak_thresh") = 30.0f,
             py::arg("strong_thresh") = 60.0f)
        .def("add_template", &PyDetector::add_template,
             py::arg("image_path"),
             py::arg("class_id"),
             py::arg("mask_path") = "",
             py::arg("num_features") = 0)
        .def("add_rotated_template", &PyDetector::add_rotated_template,
             py::arg("class_id"),
             py::arg("zero_id"),
             py::arg("theta"),
             py::arg("center_x"),
             py::arg("center_y"))
        .def("match", &PyDetector::match,
             py::arg("image_path"),
             py::arg("threshold"),
             py::arg("class_ids") = std::vector<std::string>{},
             py::arg("mask_path") = "")
        .def("get_template_info", &PyDetector::get_template_info,
             py::arg("class_id"),
             py::arg("template_id"),
             py::arg("pyramid_level") = 0)
        .def("read_classes", &PyDetector::read_classes,
             py::arg("class_ids"),
             py::arg("format") = "templates_%s.yml.gz")
        .def("write_classes", &PyDetector::write_classes,
             py::arg("format") = "templates_%s.yml.gz")
        .def("num_templates", &PyDetector::num_templates);
}
