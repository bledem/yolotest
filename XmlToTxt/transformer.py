import os
import glob
from objectmapper import ObjectMapper
from reader import Reader


class Transformer(object):
    def __init__(self, xml_dir, out_dir, img_dir):
        self.xml_dir = xml_dir
        self.out_dir = out_dir
        self.img_files = glob.glob(img_dir + "/*")
        self.img_names = []
        for i in range(len(self.img_files)):
            self.img_names.append(os.path.basename(os.path.splitext(self.img_files[i])[0]))

    def transform(self):
        reader = Reader(xml_dir=self.xml_dir)
        xml_files = reader.get_xml_files()
        classes = reader.get_classes()
        object_mapper = ObjectMapper()
        annotations = object_mapper.bind_files(xml_files)
        self.write_to_txt(annotations, classes)

    def write_to_txt(self, annotations, classes):

        image_list = open("./images_list.txt", "w+")
        for annotation in annotations:
            if (annotation.filename in self.img_names):
                print("writing")
                with open(os.path.join(self.out_dir, self.darknet_filename_format(annotation.filename)), "w+") as f:
                    f.write(self.to_darknet_format(annotation, classes, image_list))
            else:
                print("not writing for",annotation.filename )

    def to_darknet_format(self, annotation, classes, image_list):
        result = []
        images = []
        done = False
        for obj in annotation.objects:
            x, y, width, height = self.get_object_params(obj, annotation.size)
            print (annotation.filename, x, y, width, height )
            result.append("%d %.2f %.2f %.2f %.2f" % (classes[obj.name], x, y, width, height))
            if done==False:
                image_list.write(annotation.filename + "\n")
                done=True
        return "\n".join(result)

    def list_images(self, annotation, classes):
        images = []
        for obj in annotation.objects:
            images.append(annotation.filename)
        return "\n".join(images)

    @staticmethod
    def get_object_params(obj, size):
        image_width = 1.0 * size.width
        image_height = 1.0 * size.height

        box = obj.box
        absolute_x = box.xmin + 0.5 * (box.xmax - box.xmin)
        absolute_y = box.ymin + 0.5 * (box.ymax - box.ymin)

        absolute_width = box.xmax - box.xmin
        absolute_height = box.ymax - box.ymin

        x = absolute_x / image_width
        y = absolute_y / image_height
        width = absolute_width / image_width
        height = absolute_height / image_height

        return x, y, width, height

    @staticmethod
    def darknet_filename_format(filename):
        pre, ext = os.path.splitext(filename)
        return "%s.txt" % pre
