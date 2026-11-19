# Source and parameter attribution

The native stateless arithmetic and model routing are translated from the
independent mathematical reference `mauri870/dlssnr-pytorch`, fixed commit
`3beef7935a8ef976b47ab78b906e75600c52d2de`. The exact source-file hashes are recorded
in `tools/reference/source.lock.json`. This notice applies to the translated
native CUDA family files, `src/features/nr/native_graph.cpp`, and the isolated
reference fixture routing in `tools/native_nr/prepare_fixture.py`.

MIT License

Copyright (c) 2026 Mauri de Souza Meneguzzo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The network weights belong to NVIDIA and are not covered by this MIT license.
They are never included in this source tree. Original bytes, provenance and
distribution terms are registered separately in public-asset receipts.

JSON parsing reuses `nlohmann/json` v3.12.0 at fixed commit
`55f93686c01528224f448c19128836e7df245f72`. Its original header, MIT license and
SHA256 source receipt accompany `third_party/nlohmann/`.

The SDR temporal candidate adapts history reconstruction and composition
from `maanHimself/OpenDLSS-NR` at commit
`9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1`, primarily
`demo/shaders/nr_common.glsl` and `demo/shaders/nr_composite.comp`.
The MIT License text above applies to this material with the copyright
notice `Copyright (c) 2026 maan`.
