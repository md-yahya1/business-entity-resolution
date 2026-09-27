"""Compact, deterministic candidate retrieval for full laptop-scale inference."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ..preprocessing import extract_address_hints, preprocess_dataframe

STOP = {
    "the","a","an","and","or","inc","incorporated","llc","ltd","limited",
    "corp","corporation","co","company","pvt","private","sa","sas","sarl",
    "gmbh","ag","spa","srl","bv","nv","oy","ab","in","at","of","to","for",
    "on","by","from"
}
GENERIC = {
    "store","shop","market","restaurant","hotel","cafe","services","service",
    "center","centre","group","holdings","enterprise","enterprises","solutions",
    "international","global","trading","street","road","avenue","drive","lane",
    "building","floor","suite","plaza","block","box","post","city","north",
    "south","east","west","new","st","rd","ave","dr"
}

def selective_token(name: str) -> str:
    for token in name.split():
        if len(token) >= 3 and token not in STOP and token not in GENERIC:
            return token
    parts = name.split()
    return parts[0] if parts else ""

def _hash_keys(country: np.ndarray, value: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"c": country, "v": value})
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64, copy=False)

@dataclass
class PostingIndex:
    hashes: np.ndarray
    rows: np.ndarray

    @classmethod
    def build(cls, keys: np.ndarray, rows: np.ndarray, cap: int) -> "PostingIndex":
        mask = keys != 0
        keys = keys[mask]
        rows = rows[mask]
        if len(keys) == 0:
            return cls(np.empty(0, np.uint64), np.empty(0, np.int32))

        order = np.argsort(keys, kind="mergesort")
        sk = keys[order]
        sr = rows[order]

        starts = np.empty(len(sk), dtype=np.bool_)
        starts[0] = True
        starts[1:] = sk[1:] != sk[:-1]
        group_start = np.maximum.accumulate(np.where(starts, np.arange(len(sk)), 0))
        keep = (np.arange(len(sk)) - group_start) < cap
        return cls(sk[keep], sr[keep].astype(np.int32, copy=False))

    def lookup(self, key: np.uint64) -> np.ndarray:
        if key == 0 or len(self.hashes) == 0:
            return np.empty(0, np.int32)
        left = np.searchsorted(self.hashes, key, side="left")
        right = np.searchsorted(self.hashes, key, side="right")
        return self.rows[left:right]

class CompactCandidateIndex:
    """Sorted uint64 hash postings; no per-key Python list/dict objects."""

    VERSION = 3

    def __init__(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                 posting_cap: int = 64, exact_cap: int = 32):
        t0 = time.time()
        s = pd.concat([
            preprocess_dataframe(s2_df),
            preprocess_dataframe(s3_df)
        ], ignore_index=True)

        self.ids = s["entity_id"].astype(str).to_numpy()
        self.names = s["business_name_normalized"].fillna("").astype(str).to_numpy()
        self.addresses = s["business_address_normalized"].fillna("").astype(str).to_numpy()
        self.countries = s["country_normalized"].fillna("").astype(str).to_numpy()
        self.name_len = np.fromiter((len(x) for x in self.names), dtype=np.int16, count=len(s))
        self.addr_len = np.fromiter((len(x) for x in self.addresses), dtype=np.int32, count=len(s))

        hints = [extract_address_hints(x) for x in self.addresses]
        self.postal = np.array([h["postal_code"] for h in hints], dtype=object)
        self.house = np.array([h["house_number"] for h in hints], dtype=object)
        self.city = np.array([h["city"] for h in hints], dtype=object)
        self.state = np.array([h["state"] for h in hints], dtype=object)
        self.address_token = np.array(
            [selective_token(x) for x in self.addresses], dtype=object
        )

        n = len(s)
        rows = np.arange(n, dtype=np.int32)

        self.exact_name = PostingIndex.build(
            _hash_keys(self.countries, self.names), rows, exact_cap
        )
        self.prefix3 = PostingIndex.build(
            _hash_keys(self.countries, np.array([x[:3] for x in self.names], dtype=object)),
            rows, posting_cap
        )
        self.first_token = PostingIndex.build(
            _hash_keys(self.countries, np.array([selective_token(x) for x in self.names], dtype=object)),
            rows, posting_cap
        )
        self.postal_index = PostingIndex.build(
            _hash_keys(self.countries, self.postal), rows, posting_cap
        )
        self.house_index = PostingIndex.build(
            _hash_keys(self.countries, self.house), rows, posting_cap
        )
        self.city_index = PostingIndex.build(
            _hash_keys(self.countries, self.city), rows, posting_cap
        )
        # Address-token blocking is a cheap second identity signal. It helps
        # recover records whose business names changed substantially but whose
        # location text still shares a distinctive token.
        self.address_token_index = PostingIndex.build(
            _hash_keys(self.countries, self.address_token), rows, posting_cap
        )
        # Exact address is a strong identity signal when the business name changed.
        self.address_exact_index = PostingIndex.build(
            _hash_keys(self.countries, self.addresses), rows, exact_cap
        )
        # Postal + house is a cheap complementary location key.
        self.postal_house_index = PostingIndex.build(
            _hash_keys(
                self.countries,
                np.array([p + "|" + h for p, h in zip(self.postal, self.house)], dtype=object),
            ), rows, posting_cap
        )

        self._hash_country_name = _hash_keys(self.countries, self.names)
        self._hash_country_prefix3 = _hash_keys(
            self.countries, np.array([x[:3] for x in self.names], dtype=object)
        )
        self._hash_country_postal = _hash_keys(self.countries, self.postal)
        self._hash_country_house = _hash_keys(self.countries, self.house)
        self._hash_country_city = _hash_keys(self.countries, self.city)
        self._hash_country_address_token = _hash_keys(
            self.countries, self.address_token
        )
        self._hash_country_address = _hash_keys(self.countries, self.addresses)
        self._hash_country_postal_house = _hash_keys(
            self.countries,
            np.array([p + "|" + h for p, h in zip(self.postal, self.house)], dtype=object),
        )
        self._hash_country_state = _hash_keys(self.countries, self.state)

        self.n = n
        self.build_seconds = time.time() - t0

    def candidates(self, country: str, name: str, address: str,
                   postal: str, house: str, city: str, state: str,
                   limit: int = 10, hashes=None) -> np.ndarray:
        if not country:
            return np.empty(0, np.int32)

        if hashes is None:
            values = np.array([
                name, name[:3], selective_token(name), postal, house, city,
                selective_token(address), address, postal + "|" + house
            ], dtype=object)
            hashes = pd.util.hash_pandas_object(
                pd.DataFrame({"c": np.repeat(country, len(values)), "v": values}), index=False
            ).to_numpy(dtype=np.uint64)
        h_name, h_prefix, h_token, h_postal, h_house, h_city, h_address_token, h_address, h_postal_house = hashes
        blocks = []
        if name:
            blocks.append(self.exact_name.lookup(h_name))
            if len(name) >= 3:
                blocks.append(self.prefix3.lookup(h_prefix))
            if selective_token(name):
                blocks.append(self.first_token.lookup(h_token))
        if postal:
            blocks.append(self.postal_index.lookup(h_postal))
        if house:
            blocks.append(self.house_index.lookup(h_house))
        if city:
            blocks.append(self.city_index.lookup(h_city))
        address_token = selective_token(address)
        if address_token:
            blocks.append(self.address_token_index.lookup(h_address_token))
        if address:
            blocks.append(self.address_exact_index.lookup(h_address))
        if postal and house:
            blocks.append(self.postal_house_index.lookup(h_postal_house))

        blocks = [x for x in blocks if len(x)]
        if not blocks:
            return np.empty(0, np.int32)

        pool = np.unique(np.concatenate(blocks))
        if len(pool) <= limit:
            return pool.astype(np.int32, copy=False)

        # Cheap deterministic pre-ranking; no fuzzy scoring here.
        score = np.zeros(len(pool), dtype=np.float32)
        cn = self.countries[pool]
        score += (cn == country) * 1000.0
        score += (self.names[pool] == name) * 500.0
        score += (self.addresses[pool] == address) * 400.0
        score += (self.postal[pool] == postal) * 250.0
        score += (self.house[pool] == house) * 150.0
        score += (self.city[pool] == city) * 100.0
        score += (self.address_token[pool] == address_token) * 120.0
        score += (self.state[pool] == state) * 50.0
        if name:
            score += (self._hash_country_prefix3[pool] == h_prefix) * 20.0
        score -= np.abs(self.name_len[pool] - len(name)).astype(np.float32) * 0.5
        score -= np.abs(self.addr_len[pool] - len(address)).astype(np.float32) * 0.05

        take = np.argpartition(-score, min(limit, len(pool)) - 1)[:limit]
        # Stable deterministic order after the numeric top-k.
        selected = pool[take]
        order = np.lexsort((self.ids[selected], -score[take]))
        return selected[order].astype(np.int32, copy=False)

def build_index(s2_df: pd.DataFrame, s3_df: pd.DataFrame, **kwargs) -> CompactCandidateIndex:
    return CompactCandidateIndex(s2_df, s3_df, **kwargs)
