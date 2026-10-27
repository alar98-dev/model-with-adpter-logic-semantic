# ==============================================================================
# 🧬 SWA-TensorProtocol v1: Intermediate Tensor Communication Layer
# Protocolo de Comunicação Tensorial Intermediária via Base64 & Assinaturas BLAKE3
# ==============================================================================

import base64
import binascii
import struct
import time
import json
import blake3
import numpy as np
from dataclasses import dataclass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TensorIntegrityError(Exception):
    """Exceção para violação de integridade ou divergência da assinatura BLAKE3."""
    pass


class TensorPacketError(ValueError):
    """Pacote UDP inválido, truncado ou fora dos limites operacionais."""


@dataclass(frozen=True)
class UdpTensorPacket:
    """Envelope validado do transporte binário SWA-TensorProtocol/v2."""

    sequence: int
    timestamp_ns: int
    tensor: np.ndarray
    metadata: dict
    flags: int = 0


class UdpTensorProtocol:
    """Transporte binário pequeno e determinístico para um datagrama UDP.

    O v1 continua sendo usado pelo HUD local. O v2 é reservado ao barramento
    interno em ``[::1]:20011``: não usa Base64, limita cada datagrama ao MTU
    seguro e assina cabeçalho + metadados + bytes do tensor.

    Layout (network byte order)::

        magic, version, dtype, ndim, flags, reserved,
        sequence, timestamp_ns, payload_len, shape_len, metadata_len, digest
        shape_bytes, metadata_json, tensor_bytes
    """

    MAGIC = b"SWAT"
    VERSION = 2
    UDP_PORT = 20011
    MAX_DATAGRAM = 1_400
    SAFE_DATAGRAM = 1_400
    MAX_PAYLOAD = 48 * 1024
    MAX_METADATA = 4 * 1024
    MAX_DIMENSIONS = 8
    _HEADER = struct.Struct("!4sBBBBBQQIII32s")
    _DTYPES = {1: np.dtype("<f4"), 2: np.dtype("<f2")}
    _DTYPE_CODES = {v.str: k for k, v in _DTYPES.items()}
    FLAG_FRAGMENT = 1

    @classmethod
    def encode(cls, tensor, *, sequence=0, metadata=None, flags=0, dtype="float32") -> bytes:
        array = np.asarray(tensor, dtype=dtype)
        if array.ndim == 0 or array.ndim > cls.MAX_DIMENSIONS:
            raise TensorPacketError("ndim fora do limite")
        if not np.isfinite(array).all():
            raise TensorPacketError("tensor contém NaN ou infinito")
        dtype_code = cls._DTYPE_CODES.get(array.dtype.str)
        if dtype_code is None:
            raise TensorPacketError("dtype deve ser float16 ou float32")
        array = np.ascontiguousarray(array)
        payload = array.tobytes(order="C")
        if len(payload) > cls.MAX_PAYLOAD:
            raise TensorPacketError(f"payload excede {cls.MAX_PAYLOAD} bytes")

        shape = b"".join(struct.pack("!I", int(dim)) for dim in array.shape)
        meta_bytes = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")).encode("utf-8")
        if len(meta_bytes) > cls.MAX_METADATA:
            raise TensorPacketError("metadata excede o limite")
        if not isinstance(sequence, int) or sequence < 0:
            raise TensorPacketError("sequence deve ser inteiro não-negativo")
        if not 0 <= int(flags) <= 255:
            raise TensorPacketError("flags fora do intervalo uint8")

        timestamp_ns = time.time_ns()
        body = shape + meta_bytes + payload
        header_without_digest = cls._HEADER.pack(
            cls.MAGIC, cls.VERSION, dtype_code, array.ndim, int(flags), 0,
            sequence, timestamp_ns, len(payload), len(shape), len(meta_bytes), b"\0" * 32,
        )
        digest = blake3.blake3(header_without_digest + body).digest()
        packet = cls._HEADER.pack(
            cls.MAGIC, cls.VERSION, dtype_code, array.ndim, int(flags), 0,
            sequence, timestamp_ns, len(payload), len(shape), len(meta_bytes), digest,
        ) + body
        if len(packet) > cls.MAX_DATAGRAM:
            raise TensorPacketError("datagrama excede o limite UDP seguro")
        return packet

    @classmethod
    def encode_fragments(cls, tensor, *, sequence=0, metadata=None, dtype="float32") -> list[bytes]:
        """Divide um tensor em datagramas UDP pequenos o bastante para o MTU local.

        Cada fragmento é assinado individualmente e carrega a forma original,
        índice e cardinalidade do conjunto. O receptor Rust só publica no OODA
        depois de recompor todos os fragmentos da mesma sequência.
        """
        array = np.asarray(tensor, dtype=dtype)
        if array.ndim == 0 or array.ndim > cls.MAX_DIMENSIONS:
            raise TensorPacketError("ndim fora do limite")
        if not np.isfinite(array).all():
            raise TensorPacketError("tensor contém NaN ou infinito")
        if array.dtype.str not in cls._DTYPE_CODES:
            raise TensorPacketError("dtype deve ser float16 ou float32")
        array = np.ascontiguousarray(array)
        flat = array.reshape(-1)
        if flat.size == 0:
            raise TensorPacketError("tensor vazio")

        base_metadata = dict(metadata or {})
        base_metadata.update({
            "fragmented": True,
            "tensor_shape": [int(dim) for dim in array.shape],
        })
        # Reserva espaço para os campos de fragmentação e mantém margem para
        # pequenas variações de metadata sem aproximar o limite IP.
        metadata_budget = 240
        chunk_bytes = cls.SAFE_DATAGRAM - cls._HEADER.size - 4 - metadata_budget
        items_per_fragment = max(1, chunk_bytes // array.dtype.itemsize)
        fragment_count = (flat.size + items_per_fragment - 1) // items_per_fragment
        if fragment_count > 256:
            raise TensorPacketError("tensor excede o limite de fragmentação UDP")

        packets = []
        for fragment_index in range(fragment_count):
            start = fragment_index * items_per_fragment
            end = min(flat.size, start + items_per_fragment)
            fragment_metadata = dict(base_metadata)
            fragment_metadata.update({
                "fragment_index": fragment_index,
                "fragment_count": fragment_count,
            })
            packet = cls.encode(
                flat[start:end],
                sequence=sequence,
                metadata=fragment_metadata,
                flags=cls.FLAG_FRAGMENT,
                dtype=dtype,
            )
            if len(packet) > cls.SAFE_DATAGRAM:
                raise TensorPacketError("fragmento excede o MTU seguro")
            packets.append(packet)
        return packets

    @classmethod
    def decode(cls, datagram: bytes) -> UdpTensorPacket:
        if not isinstance(datagram, (bytes, bytearray, memoryview)):
            raise TensorPacketError("datagrama deve ser bytes")
        raw = bytes(datagram)
        if len(raw) < cls._HEADER.size:
            raise TensorPacketError("datagrama truncado")
        if len(raw) > cls.MAX_DATAGRAM:
            raise TensorPacketError("datagrama excede o limite")

        (magic, version, dtype_code, ndim, flags, reserved, sequence, timestamp_ns,
         payload_len, shape_len, metadata_len, received_digest) = cls._HEADER.unpack_from(raw)
        if magic != cls.MAGIC or version != cls.VERSION:
            raise TensorPacketError("magic ou versão incompatível")
        if reserved != 0 or dtype_code not in cls._DTYPES:
            raise TensorPacketError("cabeçalho inválido")
        if not 1 <= ndim <= cls.MAX_DIMENSIONS or shape_len != ndim * 4:
            raise TensorPacketError("shape inválido")
        if payload_len <= 0 or payload_len > cls.MAX_PAYLOAD or metadata_len > cls.MAX_METADATA:
            raise TensorPacketError("tamanho de payload/metadata inválido")
        body_start = cls._HEADER.size
        body_len = shape_len + metadata_len + payload_len
        if body_start + body_len != len(raw):
            raise TensorPacketError("comprimento declarado não corresponde ao datagrama")

        header_without_digest = bytearray(raw[:cls._HEADER.size])
        digest_offset = cls._HEADER.size - 32
        header_without_digest[digest_offset:] = b"\0" * 32
        expected_digest = blake3.blake3(bytes(header_without_digest) + raw[body_start:]).digest()
        if received_digest != expected_digest:
            raise TensorIntegrityError("assinatura BLAKE3 do datagrama diverge")

        shape_raw = raw[body_start:body_start + shape_len]
        shape = tuple(struct.unpack("!" + "I" * ndim, shape_raw))
        if any(dim <= 0 for dim in shape):
            raise TensorPacketError("dimensão inválida")
        dtype = cls._DTYPES[dtype_code]
        expected_payload = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected_payload != payload_len:
            raise TensorPacketError("payload não corresponde ao shape/dtype")
        meta_start = body_start + shape_len
        try:
            metadata = json.loads(raw[meta_start:meta_start + metadata_len].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TensorPacketError("metadata JSON inválido") from exc
        if not isinstance(metadata, dict):
            raise TensorPacketError("metadata deve ser objeto JSON")
        tensor_start = meta_start + metadata_len
        tensor = np.frombuffer(raw[tensor_start:], dtype=dtype).reshape(shape).copy()
        if not np.isfinite(tensor).all():
            raise TensorPacketError("tensor contém NaN ou infinito")
        return UdpTensorPacket(sequence, timestamp_ns, tensor, metadata, flags)


class SWATensorProtocol:
    """
    Protocolo de empacotamento, serialização e validação imunológica de tensores.
    Transforma tensores (NumPy/PyTorch) em pacotes seguros codificados em Base64,
    assinados digitalmente via hash BLAKE3 para comunicação entre LLMs, MCP e runtime SWA.
    """

    PROTOCOL_VERSION = "SWA-TensorProtocol/v1"

    @classmethod
    def encode(cls, tensor, metadata: dict = None, dtype: str = "float32") -> dict:
        """
        Serializa um tensor (NumPy ou PyTorch) para um dicionário JSON-safe com assinatura BLAKE3.

        :param tensor: np.ndarray ou torch.Tensor
        :param metadata: Dicionário opcional com parâmetros semânticos (ex: fase, entropia)
        :param dtype: 'float32', 'float16' ou 'float64'
        :return: dict contendo o pacote tensorial codificado e assinado
        """
        # Converte torch.Tensor para numpy se necessário
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            tensor_np = tensor
        elif isinstance(tensor, (list, tuple)):
            tensor_np = np.array(tensor, dtype=np.float32)
        else:
            raise ValueError(f"Tipo de tensor não suportado: {type(tensor)}")

        # Ajusta dtype
        if dtype == "float16":
            tensor_np = tensor_np.astype(np.float16)
        elif dtype == "float32":
            tensor_np = tensor_np.astype(np.float32)
        elif dtype == "float64":
            tensor_np = tensor_np.astype(np.float64)

        raw_bytes = tensor_np.tobytes()
        b64_str = base64.b64encode(raw_bytes).decode('ascii')
        shape = list(tensor_np.shape)

        # Cálculo da Assinatura Topológica BLAKE3
        hasher = blake3.blake3()
        hasher.update(b64_str.encode('utf-8'))
        hasher.update(json.dumps(shape).encode('utf-8'))
        hasher.update(dtype.encode('utf-8'))
        blake_sig = hasher.hexdigest()

        packet = {
            "protocol": cls.PROTOCOL_VERSION,
            "shape": shape,
            "dtype": dtype,
            "encoding": "base64",
            "blake3_signature": blake_sig,
            "tensor_b64": b64_str,
            "metadata": metadata or {},
            "timestamp": time.time()
        }

        return packet

    @classmethod
    def decode(cls, packet: dict, return_type: str = "numpy"):
        """
        Decodifica um pacote tensorial e verifica rigorosamente a assinatura BLAKE3.

        :param packet: dict no formato SWA-TensorProtocol/v1
        :param return_type: 'numpy' ou 'torch'
        :return: np.ndarray ou torch.Tensor decodificado
        """
        if packet.get("protocol") != cls.PROTOCOL_VERSION:
            raise ValueError(f"Versão de protocolo incompatível: {packet.get('protocol')}")

        shape = packet.get("shape")
        dtype_str = packet.get("dtype", "float32")
        b64_str = packet.get("tensor_b64", "")
        received_sig = packet.get("blake3_signature", "")

        # Verificação da Assinatura BLAKE3 (Integridade Imunológica)
        hasher = blake3.blake3()
        hasher.update(b64_str.encode('utf-8'))
        hasher.update(json.dumps(shape).encode('utf-8'))
        hasher.update(dtype_str.encode('utf-8'))
        expected_sig = hasher.hexdigest()

        if received_sig != expected_sig:
            raise TensorIntegrityError(
                f"🚨 [SWA TENSOR PROTOCOL] FALHA DE INTEGRIDADE: Assinatura BLAKE3 divergente! "
                f"Recebido: {received_sig} | Esperado: {expected_sig}"
            )

        # Decodificação Base64 -> Raw Bytes -> NumPy
        raw_bytes = base64.b64decode(b64_str.encode('ascii'))
        np_dtype = getattr(np, dtype_str, np.float32)
        array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)

        if return_type == "torch":
            if not HAS_TORCH:
                raise RuntimeError("PyTorch não disponível neste ambiente.")
            return torch.from_numpy(array.copy())

        return array


# --- Teste Rápido de Unidade ---
if __name__ == "__main__":
    print("🧪 Testando SWA-TensorProtocol...")
    vec = np.random.randn(1, 2560).astype(np.float32)
    meta = {"phase_theta": 1.414, "entropy": 0.05}
    
    pkt = SWATensorProtocol.encode(vec, metadata=meta)
    print(f"✅ Pacote Codificado! Assinatura BLAKE3: {pkt['blake3_signature']}")
    print(f"  Tamanho Base64: {len(pkt['tensor_b64'])} caracteres")
    
    restored = SWATensorProtocol.decode(pkt)
    diff = np.abs(vec - restored).max()
    print(f"✅ Pacote Decodificado! Erro Máximo de Reconstrução: {diff}")
    assert diff == 0.0, "Divergência de reconstrução!"
    print("🎉 Teste de protocolo concluído com sucesso!")
