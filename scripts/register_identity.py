#!/usr/bin/env python3
"""CLI for registering facial identities into the SWA-Aeon identity database.

Usage:
    python register_identity.py --name "ALifer" --photo foto.jpg
    python register_identity.py --name "Joao" --photo foto.jpg --apelido "JP" --email "joao@example.com" --role admin
    python register_identity.py --list
    python register_identity.py --add-angle --name "ALifer" --photo foto_lado.jpg --angle left
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swa.db.identity import IdentityController


def load_face_encoding(photo_path):
    """Load an image and extract the first face encoding (128D)."""
    import cv2
    import face_recognition
    import numpy as np

    image = face_recognition.load_image_file(photo_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print(f"[ERRO] Nenhuma face detectada em: {photo_path}")
        sys.exit(1)
    if len(encodings) > 1:
        print(f"[AVISO] {len(encodings)} faces detectadas. Usando a primeira.")

    return encodings[0]


def register_identity(args):
    db = IdentityController()

    existing = db.list_identities()
    for ident in existing:
        if ident.get("nome") == args.name:
            print(f"[INFO] Identidade '{args.name}' já existe (ID: {ident['id']}).")
            if args.add_angle:
                print(f"  Adicionando ângulo '{args.angle}'...")
                encoding = load_face_encoding(args.photo)
                db.add_face_encoding(
                    identity_id=ident["id"],
                    embedding_128=encoding.tolist(),
                    angle=args.angle,
                    photo_path=os.path.abspath(args.photo),
                )
                print(f"  Ângulo '{args.angle}' registrado com sucesso.")
            else:
                print("  Use --add-angle para adicionar uma nova visão.")
            return

    encoding = load_face_encoding(args.photo)

    ident = db.create_identity(
        nome=args.name,
       apelido=args.apelido,
        email=args.email,
        papel=args.role or "usuario",
        metadata={"registered_via": "cli", "photo_path": os.path.abspath(args.photo)},
    )

    db.add_face_encoding(
        identity_id=ident["id"],
        embedding_128=encoding.tolist(),
        angle=args.angle or "front",
        photo_path=os.path.abspath(args.photo),
    )

    print(f"\n[OK] Identidade registrada:")
    print(f"  Nome:    {args.name}")
    print(f"  ID:      {ident['id']}")
    print(f"  Papel:   {args.role or 'usuario'}")
    print(f"  Ângulo:  {args.angle or 'front'}")
    print(f"  Foto:    {args.photo}")


def list_identities(args):
    db = IdentityController()
    identities = db.list_identities()

    if not identities:
        print("[INFO] Nenhuma identidade registrada.")
        return

    print(f"\n{'='*70}")
    print(f"  IDENTIDADES REGISTRADAS ({len(identities)} total)")
    print(f"{'='*70}")
    for ident in identities:
        faces = db.get_face_encodings(ident["id"])
        angles = [f.get("angle", "?") for f in faces]
        print(f"\n  [{ident['id'][:8]}] {ident['nome']}")
        if ident.get("apelido"):
            print(f"    Apelido: {ident['apelido']}")
        if ident.get("email"):
            print(f"    Email:   {ident['email']}")
        print(f"    Papel:   {ident.get('papel', 'usuario')}")
        print(f"    Ângulos: {', '.join(angles) if angles else 'nenhum'}")
        print(f"    Criado:  {ident.get('created_at', '?')}")
    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="SWA-Aeon Identity Registration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--name", help="Nome da identidade")
    parser.add_argument("--photo", help="Caminho para foto de referência (JPEG/PNG)")
    parser.add_argument("--apelido", help="Apelido (opcional)")
    parser.add_argument("--email", help="Email (opcional)")
    parser.add_argument("--role", choices=["admin", "usuario", "visitante"], default="usuario",
                        help="Papel da identidade (default: usuario)")
    parser.add_argument("--angle", choices=["front", "left", "right"], default="front",
                        help="Ângulo da foto (default: front)")
    parser.add_argument("--add-angle", action="store_true",
                        help="Adicionar novo ângulo para identidade existente")
    parser.add_argument("--list", action="store_true", help="Listar todas as identidades")
    parser.add_argument("--history", help="Mostrar histórico de reconhecimento (nome)")

    args = parser.parse_args()

    if args.list:
        list_identities(args)
    elif args.name and args.photo:
        register_identity(args)
    elif args.history:
        db = IdentityController()
        identities = db.list_identities()
        target = None
        for ident in identities:
            if ident.get("nome") == args.history or ident.get("apelido") == args.history:
                target = ident
                break
        if target:
            history = db.get_recognition_history(target["id"], limit=20)
            print(f"\nHistórico de reconhecimento para {target['nome']}:")
            for h in history:
                print(f"  {h.get('detected_at', '?')} | conf={h.get('confidence', 0):.2f} | angle={h.get('angle', '?')}")
        else:
            print(f"[ERRO] Identidade '{args.history}' não encontrada.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
