from __future__ import annotations

import json
from pathlib import Path
import shutil

import tsplib_engine as E

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "_external" / "tsplib_net" / "TSPLib.Net-master" / "TSPLIB95" / "tsp"
TARGET = HERE / "data_tsplib_scale"
STRATA = {
    "n_040_120": "eil51,berlin52,st70,eil76,pr76,gr96,rat99,rd100,kroB100,kroC100,kroD100,kroE100,eil101,lin105,pr107",
    "n_121_249": "pr124,bier127,ch130,pr136,gr137,pr144,kroA150,kroB150,pr152,u159,rat195,d198,gr202,kroA200,kroB200,tsp225,ts225,pr226,gr229",
    "n_250_499": "gil262,pr264,a280,pr299,lin318,linhp318,rd400,fl417,gr431,pr439,pcb442,d493",
    "n_500_999": "ali535,u574,rat575,p654,d657,gr666,u724,rat783",
    "n_1000_2500": "pr1002,u1060,vm1084,pcb1173,d1291,rl1304,rl1323,nrw1379,fl1400,u1432,fl1577,d1655,vm1748,u1817,d2103,u2152,u2319,pr2392",
}
BKS = {
    "eil51": 426, "berlin52": 7542, "st70": 675, "eil76": 538,
    "pr76": 108159, "gr96": 55209, "rat99": 1211, "rd100": 7910,
    "kroB100": 22141, "kroC100": 20749, "kroD100": 21294, "kroE100": 22068,
    "eil101": 629, "lin105": 14379, "pr107": 44303, "pr124": 59030,
    "bier127": 118282, "ch130": 6110, "pr136": 96772, "gr137": 69853,
    "pr144": 58537, "kroA150": 26524, "kroB150": 26130, "pr152": 73682,
    "u159": 42080, "rat195": 2323, "d198": 15780, "gr202": 40160,
    "kroA200": 29368, "kroB200": 29437, "tsp225": 3916, "ts225": 126643,
    "pr226": 80369, "gr229": 134602, "gil262": 2378, "pr264": 49135,
    "a280": 2579, "pr299": 48191, "lin318": 42029, "linhp318": 41345,
    "rd400": 15281, "fl417": 11861, "gr431": 171414, "pr439": 107217,
    "pcb442": 50778, "d493": 35002, "ali535": 202339, "u574": 36905,
    "rat575": 6773, "p654": 34643, "d657": 48912, "gr666": 294358,
    "u724": 41910, "rat783": 8806, "pr1002": 259045, "u1060": 224094,
    "vm1084": 239297, "pcb1173": 56892, "d1291": 50801, "rl1304": 252948,
    "rl1323": 270199, "nrw1379": 56638, "fl1400": 20127, "u1432": 152970,
    "fl1577": 22249, "d1655": 62128, "vm1748": 336556, "u1817": 57201,
    "d2103": 80450, "u2152": 64253, "u2319": 234256, "pr2392": 378032,
}


def read_tour(path: Path) -> list[int]:
    tour = []
    active = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        value = line.strip()
        upper = value.upper()
        if upper.startswith("TOUR_SECTION"):
            active = True
            continue
        if not active:
            continue
        if upper.startswith("EOF"):
            break
        for token in value.split():
            if token == "-1":
                active = False
                break
            tour.append(int(token) - 1)
        if not active:
            break
    return tour


def main() -> None:
    TARGET.mkdir(exist_ok=True)
    catalog = {}
    errors = []
    for stratum, joined in STRATA.items():
        for name in joined.split(","):
            tsp = SOURCE / f"{name}.tsp"
            opt = SOURCE / f"{name}.opt.tour"
            if not tsp.exists():
                errors.append(f"{name}: missing instance")
                continue
            header, coords, edge_type = E.parse_tsp(tsp)
            if edge_type not in ("EUC_2D", "GEO"):
                errors.append(f"{name}: unsupported {edge_type}")
                continue
            inst = E.make_instance(coords, edge_type, force_sparse=True)[0]
            bks = BKS[name]
            reference = "TSPLIB95 documented best-known tour length"
            if opt.exists():
                tour = read_tour(opt)
                try:
                    inst.validate_tour(tour)
                except ValueError as exc:
                    errors.append(f"{name}: invalid optimal tour: {exc}")
                    continue
                tour_cost = inst.tour_cost(tour)
                if tour_cost != bks:
                    errors.append(f"{name}: documented BKS {bks} != tour cost {tour_cost}")
                    continue
                reference = "TSPLIB optimal tour, cost recomputed and cross-checked"
            expected_n = int(header["DIMENSION"])
            if len(coords) != expected_n:
                errors.append(f"{name}: dimension mismatch")
                continue
            shutil.copyfile(tsp, TARGET / f"{name}.tsp")
            catalog[name] = {
                "n": expected_n, "bks": bks, "edge_weight_type": edge_type,
                "stratum": stratum, "reference": reference,
            }
    if errors:
        raise RuntimeError("\n".join(errors))
    payload = {
        "source": "https://github.com/pdrozdowski/TSPLib.Net/tree/master/TSPLIB95/tsp",
        "upstream": "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/",
        "n_instances": len(catalog),
        "strata": {key: value.split(",") for key, value in STRATA.items()},
        "instances": catalog,
    }
    (TARGET / "catalog.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"n_instances": len(catalog), "counts": {
        key: sum(row["stratum"] == key for row in catalog.values()) for key in STRATA
    }}, indent=2))


if __name__ == "__main__":
    main()
