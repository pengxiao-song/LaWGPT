import re
import json


class read_lawfile:
    def __init__(self, chapter_moder=r"第[零一二三四五六七八九十百千万]+章 .+\b", entry_mode=r"第[零一二三四五六七八九十百千万]+条\b"):
        # 识别章和节
        self.chapter_mode = chapter_moder
        self.entry_mode = entry_mode

    def read_file(self, file_path):
        # 读取文件
        self.law = {}
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
            content = content.replace("\n\n", "\n")
            content = content.replace("##", "")
            # print(content)
            chapter_p = re.search(self.chapter_mode, content)
            while chapter_p is not None:
                c_start = chapter_p.start()
                c_end = chapter_p.end()
                key = content[c_start:c_end]
                content = content[c_end:]

                chapter_p = re.search(self.chapter_mode, content)
                if chapter_p is not None:
                    end = chapter_p.start()
                    c_content = content[:end]
                    self.law[key] = self.read_entrys(c_content)
                # print(content[c_start:c_end])
                else:
                    self.law[key] = self.read_entrys(content)
        return self.law

    def read_entrys(self, content):
        entrys = {}
        entry_p = re.search(self.entry_mode, content)
        while entry_p is not None:
            e_start = entry_p.start()
            e_end = entry_p.end()
            key = content[e_start:e_end]
            content = content[e_end+1:]

            entry_p = re.search(self.entry_mode, content)
            if entry_p is not None:
                end = entry_p.start()
                e_content = content[:end]
                entrys[key] = e_content
            else:
                entrys[key] = content
        return entrys
    # entry_p = re.search(entry_mode, content)
    # while entry_p is not None:
    #     start = entry_p.start()
    #     end = entry_p.end()
    #     # print(content[start:end])
    #     content = content[end:]
    #     law[content[start:end]] = read_entrys(content)
    #     chapter_p = re.search(chapter_mode, content)

    def show(self):
        for key in self.law:
            print(key, '\n')
            for item in self.law[key]:
                print(item, ' ', self.law[key][item])


if __name__ == '__main__':
    file_path = "D:/11496/Documents/project/Laws-master/经济法/价格法(1997-12-29).md"
    r = read_lawfile()
    dict = r.read_file(file_path)
    r.show()
    print(dict)
    with open('./a.json', 'w') as f:
        # json.dumps(dict, f, ensure_ascii=False)
        json.dump(dict, f, ensure_ascii=False)
