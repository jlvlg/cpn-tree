import copy
import itertools
import xml.dom.minidom as dom
import xml.etree.ElementTree as ET
from typing import Literal, Optional, Sequence, TypedDict

type PosAttr = tuple[float, float]


class FillAttr(TypedDict):
    colour: str
    pattern: str
    filled: bool


class LineAttr(TypedDict):
    colour: str
    thick: int
    type: str


class TextAttr(TypedDict):
    colour: str
    bold: bool


class Port(TypedDict):
    main_page_place: str
    subpage_place: str


type Size = tuple[float, float]


class ArrowAttr(TypedDict):
    headsize: float
    currentcyckle: int


class CPN:
    def __init__(self) -> None:
        self.root = ET.Element("workspaceElements")
        self.counter = itertools.count(start=1)
        self.__insert(
            parent=self.root,
            tag="generator",
            attrib={"tool": "CPN Tools", "version": "4.0.1", "format": "6"},
        )
        self.net = self.__insert(parent=self.root, tag="cpnet")
        self.declarations = self.__insert(parent=self.net, tag="globbox")
        self.declaration_blocks: dict[str, ET.Element] = {}
        self.new_declaration_block(block="std", text="Standard declarations")
        self.new_declaration_block(block="custom", text="Custom declarations")
        self.new_declaration_block(block="var", text="Variables")
        self.new_declaration_block(block="const", text="Constants")
        self.new_color(name="UNIT", type_="unit", block="std")
        self.new_color(name="BOOL", type_="bool", block="std")
        self.new_color(name="INT", type_="int", block="std")
        self.new_color(name="INTINF", type_="intinf", block="std")
        self.new_color(name="TIME", type_="time", block="std")
        self.new_color(name="REAL", type_="real", block="std")
        self.new_color(name="STRING", type_="string", block="std")
        self.__insert(self.net, "options")
        self.pages: dict[str, ET.Element] = {}
        self.places: dict[str, dict[str, ET.Element]] = {}
        self.trans: dict[str, dict[str, ET.Element]] = {}
        self.instances_node = self.__insert(parent=self.net, tag="instances")
        self.instances: dict[str, list[ET.Element]] = {}
        self.instances_trans_page: dict[str, str] = {}

    @property
    def next_id(self):
        return f"ID{next(self.counter)}"

    def find(self, id: str):
        return self.net.findall(f".//place[@id='{id}']")[0]

    def __insert(
        self,
        parent: ET.Element,
        tag: str,
        text: str = "",
        attrib: dict[str, str] = {},
    ):
        el = ET.SubElement(parent, tag, attrib)
        el.text = text
        return el

    def __insert_with_id(
        self,
        parent: ET.Element,
        tag: str,
        text: str = "",
        attrib: dict[str, str] = {},
    ):
        id = self.next_id
        return self.__insert(
            parent=parent, tag=tag, text=text, attrib={"id": id, **attrib}
        )

    def __get_pos(self, element: ET.Element):
        pos = element.find("posattr")
        if pos is None:
            raise ValueError("Element has no posattr")
        x, y = pos.get("x"), pos.get("y")
        return float(x or 0), float(y or 0)

    def new_declaration_block(self, block: str, text: str):
        block_node = self.__insert_with_id(parent=self.declarations, tag="block")
        self.declaration_blocks[block] = block_node
        self.__insert(parent=block_node, tag="id", text=text)
        return block_node

    def new_color(self, name: str, type_: str | dict[str, str], block: str = "custom"):
        color = self.__insert_with_id(
            parent=self.declaration_blocks[block], tag="color"
        )
        self.__insert(parent=color, tag="id", text=name)
        if type(type_) == dict:
            record = self.__insert(parent=color, tag="record")
            for key, value in type_.items():
                field = self.__insert(parent=record, tag="recordfield")
                self.__insert(parent=field, tag="id", text=key)
                self.__insert(parent=field, tag="id", text=value)
        else:
            self.__insert(parent=color, tag=str(type_))
        return color

    def new_variable(self, name: str, type_: str, block: str = "var"):
        var = self.__insert_with_id(parent=self.declaration_blocks[block], tag="var")
        self.__insert(parent=var, tag="id", text=name)
        type_node = self.__insert(parent=var, tag="type")
        self.__insert(parent=type_node, tag="id", text=type_)
        return var

    def new_constant(self, ml: str, block: str = "const"):
        self.__insert_with_id(parent=self.declaration_blocks[block], tag="ml", text=ml)

    def new_page(self, name: str):
        page_node = self.__insert_with_id(parent=self.net, tag="page")
        self.pages[name] = page_node
        self.places[name] = {}
        self.trans[name] = {}
        self.__insert(parent=self.pages[name], tag="pageattr", attrib={"name": name})
        self.__insert(parent=self.pages[name], tag="constraints")
        self.__insert_toplevel_instance(name)

        return page_node

    def instantiate_page(self, page: str, trans: str, subpage: str, ports: list[Port]):
        self.__insert_instance(page=page, subpage=subpage, trans=trans)
        pos = self.__get_pos(self.trans[page][trans])
        self.__insert_subst(
            element=self.trans[page][trans],
            page=page,
            subpage=subpage,
            ports=ports,
            posattr=(pos[0] - 30, pos[1] - 20),
        )

    def new_place(
        self,
        page: str,
        posattr: PosAttr,
        name: str,
        type_: str = "UNIT",
        initmark: str = "",
        size: Size = (60, 40),
        tokenPos: PosAttr = (-10, 0),
        markingPos: PosAttr = (0, 0),
        markingHidden: bool = True,
        port: Optional[Literal["Out", "In", "I/O"]] = None,
        fillattr: FillAttr = {"colour": "White", "pattern": "", "filled": False},
        lineattr: LineAttr = {"colour": "Black", "thick": 1, "type": "Solid"},
        textattr: TextAttr = {"colour": "Black", "bold": False},
    ):
        place = self.__insert_with_id(parent=self.pages[page], tag="place")
        self.places[page][name] = place
        self.__insert_posattr(element=place, posattr=posattr)
        self.__insert_fillattr(element=place, fillattr=fillattr)
        self.__insert_lineattr(element=place, lineattr=lineattr)
        self.__insert_textattr(element=place, textattr=textattr)
        self.__insert(parent=place, tag="text", text=name)
        self.__insert(
            parent=place,
            tag="ellipse",
            attrib={"w": str(size[0]), "h": str(size[1])},
        )
        self.__insert(
            parent=place,
            tag="token",
            attrib={"x": str(tokenPos[0]), "y": str(tokenPos[1])},
        )
        self.__insert(
            parent=place,
            tag="marking",
            attrib={
                "x": str(markingPos[0]),
                "y": str(markingPos[1]),
                "hidden": str(markingHidden).lower(),
            },
        )
        self.__insert_type(
            element=place,
            type_=type_,
            posattr=(posattr[0] + 50, posattr[1] - 20),
            fillattr=fillattr,
            lineattr=lineattr,
            textattr=textattr,
        )
        self.insert_initmark(
            element=place, mark=initmark, posattr=(posattr[0] + 50, posattr[1] + 20)
        )
        if port:
            self.__insert_port(
                element=place, type_=port, posattr=(posattr[0] - 30, posattr[1] - 20)
            )
        return place

    def __insert_posattr(self, element: ET.Element, posattr: PosAttr):
        self.__insert(
            parent=element,
            tag="posattr",
            attrib={"x": str(posattr[0]), "y": str(posattr[1])},
        )

    def __insert_fillattr(self, element: ET.Element, fillattr: FillAttr):
        self.__insert(
            parent=element,
            tag="fillattr",
            attrib={
                "colour": fillattr["colour"],
                "pattern": fillattr["pattern"],
                "filled": str(fillattr["filled"]).lower(),
            },
        )

    def __insert_lineattr(self, element: ET.Element, lineattr: LineAttr):
        self.__insert(
            parent=element,
            tag="lineattr",
            attrib={
                "colour": lineattr["colour"],
                "thick": str(lineattr["thick"]),
                "type": lineattr["type"],
            },
        )

    def __insert_textattr(self, element: ET.Element, textattr: TextAttr):
        self.__insert(
            parent=element,
            tag="textattr",
            attrib={
                "colour": textattr["colour"],
                "bold": str(textattr["bold"]).lower(),
            },
        )

    def __insert_type(
        self,
        element: ET.Element,
        type_: str,
        posattr: PosAttr,
        fillattr: FillAttr = {"colour": "White", "pattern": "Solid", "filled": False},
        lineattr: LineAttr = {"colour": "Black", "thick": 0, "type": "Solid"},
        textattr: TextAttr = {"colour": "Black", "bold": False},
    ):
        type_node = self.__insert_with_id(parent=element, tag="type")
        self.__insert_posattr(element=type_node, posattr=posattr)
        self.__insert_fillattr(element=type_node, fillattr=fillattr)
        self.__insert_lineattr(element=type_node, lineattr=lineattr)
        self.__insert_textattr(element=type_node, textattr=textattr)
        self.__insert(
            parent=type_node,
            tag="text",
            text=type_,
            attrib={"tool": "CPN Tools", "version": "4.0.1"},
        )
        return type_node

    def insert_initmark(
        self,
        element: ET.Element,
        mark: str,
        posattr: PosAttr,
        fillattr: FillAttr = {"colour": "White", "pattern": "Solid", "filled": False},
        lineattr: LineAttr = {"colour": "Black", "thick": 0, "type": "Solid"},
        textattr: TextAttr = {"colour": "black", "bold": False},
    ):
        initmark_node = self.__insert_with_id(parent=element, tag="initmark")
        self.__insert_posattr(element=initmark_node, posattr=posattr)
        self.__insert_fillattr(element=initmark_node, fillattr=fillattr)
        self.__insert_lineattr(element=initmark_node, lineattr=lineattr)
        self.__insert_textattr(element=initmark_node, textattr=textattr)
        self.__insert(
            parent=initmark_node,
            tag="text",
            text=mark,
            attrib={"tool": "CPN Tools", "version": "4.0.1"},
        )
        return initmark_node

    def __insert_port(
        self,
        element: ET.Element,
        type_: Literal["Out", "In", "I/O"],
        posattr: PosAttr,
        fillattr: FillAttr = {"colour": "White", "pattern": "Solid", "filled": False},
        lineattr: LineAttr = {"colour": "Black", "thick": 0, "type": "Solid"},
        textattr: TextAttr = {"colour": "black", "bold": False},
    ):
        port_node = self.__insert_with_id(
            parent=element, tag="port", attrib={"type": type_}
        )
        self.__insert_posattr(element=port_node, posattr=posattr)
        self.__insert_fillattr(element=port_node, fillattr=fillattr)
        self.__insert_lineattr(element=port_node, lineattr=lineattr)
        self.__insert_textattr(element=port_node, textattr=textattr)

    def new_trans(
        self,
        page: str,
        posattr: PosAttr,
        name: str,
        cond: str = "",
        size: Size = (60, 40),
        binding_pos: PosAttr = (7, -3),
        cond_pos: Optional[PosAttr] = None,
        fillattr: FillAttr = {"colour": "White", "pattern": "Solid", "filled": False},
        lineattr: LineAttr = {"colour": "Black", "thick": 0, "type": "Solid"},
        textattr: TextAttr = {"colour": "black", "bold": False},
    ):
        trans = self.__insert_with_id(
            parent=self.pages[page], tag="trans", attrib={"explicit": "false"}
        )
        self.trans[page][name] = trans
        self.__insert_posattr(element=trans, posattr=posattr)
        self.__insert_fillattr(element=trans, fillattr=fillattr)
        self.__insert_lineattr(element=trans, lineattr=lineattr)
        self.__insert_textattr(element=trans, textattr=textattr)
        self.__insert(parent=trans, tag="text", text=name)
        self.__insert(
            parent=trans,
            tag="box",
            attrib={"w": str(size[0]), "h": str(size[1])},
        )
        self.__insert(
            parent=trans,
            tag="binding",
            attrib={"x": str(binding_pos[0]), "y": str(binding_pos[1])},
        )
        self.__insert_cond(
            element=trans,
            cond=cond,
            posattr=cond_pos or (posattr[0], posattr[1] + 50),
            fillattr=fillattr,
            lineattr=lineattr,
            textattr=textattr,
        )

        return trans

    def __insert_cond(
        self,
        element: ET.Element,
        cond: str,
        posattr: PosAttr,
        fillattr: FillAttr = {"colour": "White", "pattern": "Solid", "filled": False},
        lineattr: LineAttr = {"colour": "Black", "thick": 0, "type": "Solid"},
        textattr: TextAttr = {"colour": "black", "bold": False},
    ):
        cond_node = self.__insert_with_id(parent=element, tag="cond")
        self.__insert_posattr(element=cond_node, posattr=posattr)
        self.__insert_fillattr(element=cond_node, fillattr=fillattr)
        self.__insert_lineattr(element=cond_node, lineattr=lineattr)
        self.__insert_textattr(element=cond_node, textattr=textattr)
        self.__insert(
            parent=cond_node,
            tag="text",
            text=cond,
            attrib={"tool": "CPN Tools", "version": "4.0.1"},
        )
        return cond_node

    def __insert_subst(
        self,
        element: ET.Element,
        page: str,
        subpage: str,
        ports: list[Port],
        posattr: PosAttr,
        fillattr: FillAttr = {"colour": "White", "pattern": "Solid", "filled": False},
        lineattr: LineAttr = {"colour": "Black", "thick": 0, "type": "Solid"},
        textattr: TextAttr = {"colour": "black", "bold": False},
    ):
        portsock = "".join(
            f"({self.places[subpage][port['subpage_place']].get('id')},{self.places[page][port['main_page_place']].get('id')})"
            for port in ports
        )
        subst = self.__insert(
            parent=element,
            tag="subst",
            attrib={
                "subpage": str(self.pages[subpage].get("id")),
                "portsock": portsock,
            },
        )
        subpageinfo = self.__insert_with_id(
            parent=subst, tag="subpageinfo", attrib={"name": subpage}
        )
        self.__insert_posattr(element=subpageinfo, posattr=posattr)
        self.__insert_fillattr(element=subpageinfo, fillattr=fillattr)
        self.__insert_lineattr(element=subpageinfo, lineattr=lineattr)
        self.__insert_textattr(element=subpageinfo, textattr=textattr)
        return subpageinfo

    def new_arc(
        self,
        page: str,
        orientation: Literal["PTOT", "TTOP", "BOTHDIR"],
        trans: str,
        place: str,
        annot: str = "",
        annot_pos: Optional[PosAttr] = None,
        bend_points: Sequence[PosAttr] = [],
        posattr: PosAttr = (0, 0),
        fillattr: FillAttr = {"colour": "White", "pattern": "Solid", "filled": False},
        lineattr: LineAttr = {"colour": "Black", "thick": 0, "type": "Solid"},
        textattr: TextAttr = {"colour": "black", "bold": False},
        arrowattr: ArrowAttr = {"headsize": 1.2, "currentcyckle": 2},
    ):
        arc = self.__insert_with_id(
            parent=self.pages[page],
            tag="arc",
            attrib={"orientation": orientation, "order": "1"},
        )
        self.__insert_posattr(element=arc, posattr=posattr)
        self.__insert_fillattr(element=arc, fillattr=fillattr)
        self.__insert_lineattr(element=arc, lineattr=lineattr)
        self.__insert_textattr(element=arc, textattr=textattr)
        self.__insert_arrowattr(element=arc, arrowattr=arrowattr)
        self.__insert(
            parent=arc,
            tag="transend",
            attrib={"idref": str(self.trans[page][trans].get("id"))},
        )
        self.__insert(
            parent=arc,
            tag="placeend",
            attrib={"idref": str(self.places[page][place].get("id"))},
        )
        if annot_pos is None:
            trans_pos = self.__get_pos(self.trans[page][trans])
            place_pos = self.__get_pos(self.places[page][place])
            annot_pos = (
                (trans_pos[0] + place_pos[0]) / 2,
                (trans_pos[1] + place_pos[1]) / 2,
            )
        self.__insert_annot(
            element=arc,
            annot=annot,
            posattr=annot_pos,
            fillattr=fillattr,
            lineattr=lineattr,
            textattr=textattr,
        )
        for point in bend_points:
            self.__insert_bend_point(
                element=arc,
                posattr=point,
                fillattr=fillattr,
                lineattr=lineattr,
                textattr=textattr,
            )
        return arc

    def __insert_arrowattr(
        self,
        element: ET.Element,
        arrowattr: ArrowAttr = {"headsize": 1.2, "currentcyckle": 2},
    ):
        self.__insert(
            parent=element,
            tag="arrowattr",
            attrib={
                "headsize": str(arrowattr["headsize"]),
                "currentcyckle": str(arrowattr["currentcyckle"]),
            },
        )

    def __insert_annot(
        self,
        element: ET.Element,
        annot: str,
        posattr: PosAttr,
        fillattr: FillAttr = {"colour": "White", "pattern": "Solid", "filled": False},
        lineattr: LineAttr = {"colour": "Black", "thick": 0, "type": "Solid"},
        textattr: TextAttr = {"colour": "Black", "bold": False},
    ):
        annot_node = self.__insert_with_id(parent=element, tag="annot")
        self.__insert_posattr(element=annot_node, posattr=posattr)
        self.__insert_fillattr(element=annot_node, fillattr=fillattr)
        self.__insert_lineattr(element=annot_node, lineattr=lineattr)
        self.__insert_textattr(element=annot_node, textattr=textattr)
        self.__insert(
            parent=annot_node,
            tag="text",
            text=annot,
            attrib={"tool": "CPN Tools", "version": "4.0.1"},
        )
        return annot_node

    def __insert_bend_point(
        self,
        element: ET.Element,
        posattr: PosAttr,
        fillattr: FillAttr = {"colour": "White", "pattern": "Solid", "filled": False},
        lineattr: LineAttr = {"colour": "Black", "thick": 0, "type": "Solid"},
        textattr: TextAttr = {"colour": "Black", "bold": False},
    ):
        bend_node = self.__insert_with_id(parent=element, tag="bendpoint")
        self.__insert_posattr(element=bend_node, posattr=posattr)
        self.__insert_fillattr(element=bend_node, fillattr=fillattr)
        self.__insert_lineattr(element=bend_node, lineattr=lineattr)
        self.__insert_textattr(element=bend_node, textattr=textattr)
        return bend_node

    def __clone_subtree(self, source: ET.Element, destination: ET.Element):
        clone = copy.deepcopy(source)
        for element in clone.iter():
            if "id" in element.attrib:
                element.set("id", self.next_id)
        destination.append(clone)
        return clone

    def __insert_toplevel_instance(self, page: str):
        node = self.__insert_with_id(
            parent=self.instances_node,
            tag="instance",
            attrib={"page": str(self.pages[page].get("id"))},
        )
        self.instances[page] = [node]

    def __insert_instance(self, page: str, subpage: str, trans: str):
        self.instances_trans_page[str(self.trans[page][trans].get("id"))] = subpage
        subtree = None
        moved = False
        if self.instances[subpage][0].get("page"):
            subtree = self.instances[subpage].pop(0)
            subtree.attrib.pop("page")
            self.instances_node.remove(subtree)
            moved = True
        else:
            subtree = self.instances[subpage][0]

        for instance_block in self.instances[page]:
            clone = self.__clone_subtree(subtree, instance_block)
            clone.set("trans", str(self.trans[page][trans].get("id")))
            if moved:
                for el in subtree.iter():
                    if (
                        "trans" in el.attrib
                        and el
                        in self.instances[
                            self.instances_trans_page[str(el.get("trans"))]
                        ]
                    ):
                        self.instances[
                            self.instances_trans_page[str(el.get("trans"))]
                        ].remove(el)
            for el in clone.iter():
                self.instances[self.instances_trans_page[str(el.get("trans"))]].append(
                    el
                )

    def save(self, path: str):
        with open(path, "w") as f:
            dom.parseString(str(self)).writexml(f, addindent="  ", newl="\n")

    def __repr__(self) -> str:
        return ET.tostring(self.root, encoding="unicode")
