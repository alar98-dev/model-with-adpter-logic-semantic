// ═══════════════════════════════════════════════════════════════
// CIDADE TENSORIAL — Mock Estático com Workers/Cargos
// 15 Setores | 46 Empresas | Workers por empresa
// ═══════════════════════════════════════════════════════════════

const CITY_SECTORS = [
  {
    id: "S01", name: "Saúde", color: "#10b981", emoji: "🏥",
    x: 0, y: -200, z: 0, radius: 80,
    metrics: { receita: 4200, custo: 3100, saude: 0.88, entropia: 0.12, funcionarios: 340 },
    empresas: [
      {
        id: "E0101", name: "Hospital Central", x: 46, y: -201, z: -11, radius: 38, color: "#34d399",
        metrics: { receita: 1800, custo: 1400, funcionarios: 120, resiliencia: 0.9, sinergia: 0.7 },
        processos: [
          { id: "P010101", name: "Triagem", x: 37, y: -212, z: -24, metrics: { throughput: 189, latencia: 12.0, erro: 0.008, ciclo: 5 } },
          { id: "P010102", name: "Internação", x: 69, y: -182, z: -20, metrics: { throughput: 54, latencia: 13.4, erro: 0.03, ciclo: 6 } },
          { id: "P010103", name: "Cirurgia", x: 58, y: -178, z: 2, metrics: { throughput: 157, latencia: 17.3, erro: 0.007, ciclo: 2 } },
          { id: "P010104", name: "Diagnóstico", x: 39, y: -212, z: -8, metrics: { throughput: 50, latencia: 18.5, erro: 0.043, ciclo: 2 } },
          { id: "P010105", name: "Farmácia", x: 38, y: -224, z: -28, metrics: { throughput: 146, latencia: 17.9, erro: 0.057, ciclo: 6 } }
        ],
        workers: [
          { id: "W10101", nome: "Eduardo Freitas", cargo: "Médico", salario: 10541, produtividade: 0.72, skills: ["Diagn\u00f3stico", "Tratamento", "Cirurgia"], vinculo: "Triagem", horas_semanais: 40 },
          { id: "W10102", nome: "Lucas Batista", cargo: "Enfermeiro", salario: 4024, produtividade: 0.64, skills: ["Cuidados", "Medica\u00e7\u00e3o", "Triagem"], vinculo: "Internação", horas_semanais: 44 },
          { id: "W10103", nome: "Rafael Correia", cargo: "Técnico de Enfermagem", salario: 6486, produtividade: 0.65, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Cirurgia", horas_semanais: 48 },
          { id: "W10104", nome: "Bianca Farias", cargo: "Farmacêutico", salario: 8546, produtividade: 0.73, skills: ["Farmacologia", "Regula\u00e7\u00e3o"], vinculo: "Diagnóstico", horas_semanais: 48 },
          { id: "W10105", nome: "Renata Campos", cargo: "Recepcionista", salario: 8132, produtividade: 0.7, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Farmácia", horas_semanais: 48 },
          { id: "W10106", nome: "Sérgio Peixoto", cargo: "Administrador", salario: 11517, produtividade: 0.66, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Triagem", horas_semanais: 48 }
        ]
      },
      {
        id: "E0102", name: "Farmácia Vida", x: -41, y: -254, z: 1, radius: 28, color: "#6ee7b7",
        metrics: { receita: 900, custo: 600, funcionarios: 45, resiliencia: 0.85, sinergia: 0.6 },
        processos: [
          { id: "P010201", name: "Dispensação", x: -35, y: -275, z: 17, metrics: { throughput: 166, latencia: 41.3, erro: 0.01, ciclo: 5 } },
          { id: "P010202", name: "Estoque", x: -47, y: -274, z: -4, metrics: { throughput: 35, latencia: 33.7, erro: 0.026, ciclo: 5 } },
          { id: "P010203", name: "Controle", x: -16, y: -240, z: -5, metrics: { throughput: 138, latencia: 23.1, erro: 0.055, ciclo: 3 } },
          { id: "P010204", name: "Logística", x: -29, y: -252, z: 0, metrics: { throughput: 150, latencia: 37.5, erro: 0.037, ciclo: 6 } }
        ],
        workers: [
          { id: "W10201", nome: "Daniela Tavares", cargo: "Farmacêutico", salario: 2700, produtividade: 0.78, skills: ["Farmacologia", "Regula\u00e7\u00e3o"], vinculo: "Dispensação", horas_semanais: 44 },
          { id: "W10202", nome: "Eduardo Freitas", cargo: "Técnico", salario: 10343, produtividade: 0.71, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Estoque", horas_semanais: 40 },
          { id: "W10203", nome: "Fernanda Costa", cargo: "Vendedor", salario: 3341, produtividade: 0.62, skills: ["Comunica\u00e7\u00e3o", "Negocia\u00e7\u00e3o", "CRM"], vinculo: "Controle", horas_semanais: 44 }
        ]
      },
      {
        id: "E0103", name: "Clínica Bem Estar", x: 45, y: -150, z: -13, radius: 25, color: "#a7f3d0",
        metrics: { receita: 600, custo: 420, funcionarios: 35, resiliencia: 0.82, sinergia: 0.65 },
        processos: [
          { id: "P010301", name: "Consultas", x: 22, y: -161, z: -15, metrics: { throughput: 185, latencia: 17.3, erro: 0.052, ciclo: 1 } },
          { id: "P010302", name: "Exames", x: 63, y: -161, z: -17, metrics: { throughput: 165, latencia: 35.6, erro: 0.048, ciclo: 2 } },
          { id: "P010303", name: "Recepção", x: 47, y: -168, z: 1, metrics: { throughput: 62, latencia: 39.0, erro: 0.055, ciclo: 2 } },
          { id: "P010304", name: "Prontuário", x: 24, y: -172, z: -23, metrics: { throughput: 83, latencia: 35.9, erro: 0.05, ciclo: 3 } }
        ],
        workers: [
          { id: "W10301", nome: "Fernanda Costa", cargo: "Médico", salario: 5957, produtividade: 0.63, skills: ["Diagn\u00f3stico", "Tratamento", "Cirurgia"], vinculo: "Consultas", horas_semanais: 40 },
          { id: "W10302", nome: "Daniela Tavares", cargo: "Psicólogo", salario: 5073, produtividade: 0.62, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Exames", horas_semanais: 48 },
          { id: "W10303", nome: "Francisco Gomes", cargo: "Recepcionista", salario: 3729, produtividade: 0.58, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Recepção", horas_semanais: 44 }
        ]
      }
    ]
  },
  {
    id: "S02", name: "Educação", color: "#6366f1", emoji: "🎓",
    x: -280, y: 0, z: -50, radius: 72,
    metrics: { receita: 3100, custo: 2800, saude: 0.82, entropia: 0.18, funcionarios: 280 },
    empresas: [
      {
        id: "E0201", name: "UniTech", x: -327, y: -12, z: -14, radius: 40, color: "#818cf8",
        metrics: { receita: 1400, custo: 1200, funcionarios: 90, resiliencia: 0.85, sinergia: 0.8 },
        processos: [
          { id: "P020101", name: "Graduação", x: -316, y: 0, z: -33, metrics: { throughput: 200, latencia: 40.5, erro: 0.017, ciclo: 1 } },
          { id: "P020102", name: "Pós-Grad", x: -304, y: 11, z: -23, metrics: { throughput: 125, latencia: 31.4, erro: 0.027, ciclo: 3 } },
          { id: "P020103", name: "Pesquisa", x: -341, y: 0, z: -7, metrics: { throughput: 167, latencia: 48.9, erro: 0.058, ciclo: 4 } },
          { id: "P020104", name: "Extensão", x: -330, y: -11, z: -13, metrics: { throughput: 87, latencia: 40.4, erro: 0.052, ciclo: 3 } }
        ],
        workers: [
          { id: "W20101", nome: "Tatiana Nunes", cargo: "Professor", salario: 7482, produtividade: 0.84, skills: ["Did\u00e1tica", "Pesquisa", "Avalia\u00e7\u00e3o"], vinculo: "Graduação", horas_semanais: 44 },
          { id: "W20102", nome: "Ricardo Lima", cargo: "Pesquisador", salario: 10699, produtividade: 0.76, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Pós-Grad", horas_semanais: 44 },
          { id: "W20103", nome: "André Carvalho", cargo: "Coordenador", salario: 3817, produtividade: 0.79, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Pesquisa", horas_semanais: 44 },
          { id: "W20104", nome: "Bianca Farias", cargo: "Bibliotecário", salario: 7780, produtividade: 0.79, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Extensão", horas_semanais: 40 }
        ]
      },
      {
        id: "E0202", name: "Escola Solar", x: -247, y: 61, z: -100, radius: 30, color: "#a5b4fc",
        metrics: { receita: 800, custo: 700, funcionarios: 60, resiliencia: 0.7, sinergia: 0.6 },
        processos: [
          { id: "P020201", name: "Fundamental", x: -230, y: 70, z: -91, metrics: { throughput: 110, latencia: 3.7, erro: 0.032, ciclo: 5 } },
          { id: "P020202", name: "Médio", x: -224, y: 67, z: -80, metrics: { throughput: 118, latencia: 45.7, erro: 0.013, ciclo: 5 } },
          { id: "P020203", name: "Recreação", x: -264, y: 54, z: -92, metrics: { throughput: 183, latencia: 29.3, erro: 0.003, ciclo: 6 } }
        ],
        workers: [
          { id: "W20201", nome: "Daniela Vieira", cargo: "Professor", salario: 7222, produtividade: 0.82, skills: ["Did\u00e1tica", "Pesquisa", "Avalia\u00e7\u00e3o"], vinculo: "Fundamental", horas_semanais: 44 },
          { id: "W20202", nome: "Bianca Farias", cargo: "Pedagogo", salario: 11512, produtividade: 0.52, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Médio", horas_semanais: 40 },
          { id: "W20203", nome: "Eduardo Freitas", cargo: "Monitor", salario: 7653, produtividade: 0.62, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Recreação", horas_semanais: 40 }
        ]
      },
      {
        id: "E0203", name: "Idiomas Fluents", x: -311, y: 57, z: -9, radius: 20, color: "#c7d2fe",
        metrics: { receita: 450, custo: 320, funcionarios: 25, resiliencia: 0.65, sinergia: 0.5 },
        processos: [
          { id: "P020301", name: "Inglês", x: -318, y: 64, z: -12, metrics: { throughput: 111, latencia: 50.2, erro: 0.059, ciclo: 2 } },
          { id: "P020302", name: "Espanhol", x: -307, y: 67, z: -20, metrics: { throughput: 103, latencia: 11.8, erro: 0.036, ciclo: 6 } },
          { id: "P020303", name: "Francês", x: -328, y: 36, z: -12, metrics: { throughput: 111, latencia: 20.7, erro: 0.047, ciclo: 3 } }
        ],
        workers: [
          { id: "W20301", nome: "Rosângela Azevedo", cargo: "Professor", salario: 5094, produtividade: 0.65, skills: ["Did\u00e1tica", "Pesquisa", "Avalia\u00e7\u00e3o"], vinculo: "Inglês", horas_semanais: 40 },
          { id: "W20302", nome: "Rafael Correia", cargo: "Recepcionista", salario: 11548, produtividade: 0.7, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Espanhol", horas_semanais: 40 },
          { id: "W20303", nome: "Mateus Lima", cargo: "Coordenador", salario: 4358, produtividade: 0.72, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Francês", horas_semanais: 40 }
        ]
      }
    ]
  },
  {
    id: "S03", name: "Comércio", color: "#f59e0b", emoji: "🛒",
    x: 250, y: -80, z: 60, radius: 75,
    metrics: { receita: 5800, custo: 4200, saude: 0.91, entropia: 0.09, funcionarios: 520 },
    empresas: [
      {
        id: "E0301", name: "Mercearia Central", x: 301, y: -53, z: 59, radius: 35, color: "#fbbf24",
        metrics: { receita: 2200, custo: 1600, funcionarios: 150, resiliencia: 0.9, sinergia: 0.7 },
        processos: [
          { id: "P030101", name: "Caixa", x: 318, y: -28, z: 48, metrics: { throughput: 131, latencia: 57.9, erro: 0.008, ciclo: 5 } },
          { id: "P030102", name: "Estoque", x: 297, y: -72, z: 67, metrics: { throughput: 30, latencia: 31.8, erro: 0.028, ciclo: 6 } },
          { id: "P030103", name: "Reposição", x: 285, y: -52, z: 48, metrics: { throughput: 24, latencia: 28.4, erro: 0.058, ciclo: 3 } },
          { id: "P030104", name: "Delivery", x: 315, y: -34, z: 64, metrics: { throughput: 171, latencia: 5.3, erro: 0.02, ciclo: 6 } }
        ],
        workers: [
          { id: "W30101", nome: "Ana Silva", cargo: "Gerente", salario: 10522, produtividade: 0.92, skills: ["Lideran\u00e7a", "Gest\u00e3o", "Planejamento"], vinculo: "Caixa", horas_semanais: 44 },
          { id: "W30102", nome: "Isabela Castro", cargo: "Caixista", salario: 11324, produtividade: 0.73, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Estoque", horas_semanais: 44 },
          { id: "W30103", nome: "Carla Fonseca", cargo: "Repositor", salario: 11542, produtividade: 0.87, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Reposição", horas_semanais: 44 },
          { id: "W30104", nome: "Thiago Melo", cargo: "Entregador", salario: 9960, produtividade: 0.95, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Delivery", horas_semanais: 40 },
          { id: "W30105", nome: "Gustavo Barros", cargo: "Estoquista", salario: 6412, produtividade: 0.77, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Caixa", horas_semanais: 40 },
          { id: "W30106", nome: "Mateus Lima", cargo: "Comprador", salario: 9230, produtividade: 0.52, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Estoque", horas_semanais: 48 }
        ]
      },
      {
        id: "E0302", name: "SuperMart", x: 256, y: -143, z: 15, radius: 32, color: "#fcd34d",
        metrics: { receita: 1800, custo: 1300, funcionarios: 120, resiliencia: 0.85, sinergia: 0.65 },
        processos: [
          { id: "P030201", name: "Perecíveis", x: 251, y: -118, z: -2, metrics: { throughput: 80, latencia: 21.8, erro: 0.026, ciclo: 2 } },
          { id: "P030202", name: "Não-Perecíveis", x: 264, y: -142, z: 31, metrics: { throughput: 179, latencia: 47.7, erro: 0.011, ciclo: 1 } },
          { id: "P030203", name: "Hortifrúti", x: 270, y: -144, z: 34, metrics: { throughput: 179, latencia: 14.8, erro: 0.055, ciclo: 2 } }
        ],
        workers: [
          { id: "W30201", nome: "Cláudia Rezende", cargo: "Gerente", salario: 10494, produtividade: 0.92, skills: ["Lideran\u00e7a", "Gest\u00e3o", "Planejamento"], vinculo: "Perecíveis", horas_semanais: 40 },
          { id: "W30202", nome: "Priscila Ramos", cargo: "Operador", salario: 3621, produtividade: 0.61, skills: ["Opera\u00e7\u00e3o", "Controle", "Monitoramento"], vinculo: "Não-Perecíveis", horas_semanais: 48 },
          { id: "W30203", nome: "Amanda Pereira", cargo: "Repositor", salario: 7208, produtividade: 0.98, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Hortifrúti", horas_semanais: 48 },
          { id: "W30204", nome: "Helena Moura", cargo: "Caixista", salario: 3980, produtividade: 0.71, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Perecíveis", horas_semanais: 40 },
          { id: "W30205", nome: "Mateus Lima", cargo: "Conferente", salario: 4146, produtividade: 0.71, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Não-Perecíveis", horas_semanais: 48 }
        ]
      },
      {
        id: "E0303", name: "PetShop Amigo", x: 198, y: -37, z: 54, radius: 18, color: "#fde68a",
        metrics: { receita: 350, custo: 240, funcionarios: 15, resiliencia: 0.6, sinergia: 0.4 },
        processos: [
          { id: "P030301", name: "Vendas", x: 210, y: -43, z: 74, metrics: { throughput: 113, latencia: 41.6, erro: 0.028, ciclo: 3 } },
          { id: "P030302", name: "Banho/Tosa", x: 185, y: -38, z: 64, metrics: { throughput: 32, latencia: 14.6, erro: 0.035, ciclo: 5 } }
        ],
        workers: [
          { id: "W30301", nome: "Beatriz Dias", cargo: "Atendente", salario: 6661, produtividade: 0.72, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Vendas", horas_semanais: 48 },
          { id: "W30302", nome: "André Carvalho", cargo: "Tosador", salario: 2653, produtividade: 0.93, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Banho/Tosa", horas_semanais: 44 },
          { id: "W30303", nome: "Helena Moura", cargo: "Veterinário", salario: 7212, produtividade: 0.83, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Vendas", horas_semanais: 40 }
        ]
      },
      {
        id: "E0304", name: "Floricultura Pétala", x: 192, y: -23, z: 46, radius: 16, color: "#fef3c7",
        metrics: { receita: 200, custo: 150, funcionarios: 8, resiliencia: 0.5, sinergia: 0.35 },
        processos: [
          { id: "P030401", name: "Montagem", x: 216, y: -34, z: 64, metrics: { throughput: 95, latencia: 13.5, erro: 0.012, ciclo: 3 } },
          { id: "P030402", name: "Entrega", x: 210, y: 0, z: 34, metrics: { throughput: 165, latencia: 6.3, erro: 0.038, ciclo: 1 } }
        ],
        workers: [
          { id: "W30401", nome: "Isabela Castro", cargo: "Florista", salario: 7337, produtividade: 0.51, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Montagem", horas_semanais: 48 },
          { id: "W30402", nome: "Bianca Farias", cargo: "Entregador", salario: 8984, produtividade: 0.63, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Entrega", horas_semanais: 48 }
        ]
      }
    ]
  },
  {
    id: "S04", name: "Indústria", color: "#ef4444", emoji: "🏭",
    x: 200, y: 150, z: -80, radius: 70,
    metrics: { receita: 7200, custo: 5800, saude: 0.78, entropia: 0.22, funcionarios: 680 },
    empresas: [
      {
        id: "E0401", name: "MetalForge", x: 231, y: 173, z: -119, radius: 38, color: "#f87171",
        metrics: { receita: 3200, custo: 2600, funcionarios: 280, resiliencia: 0.8, sinergia: 0.7 },
        processos: [
          { id: "P040101", name: "Corte", x: 256, y: 173, z: -139, metrics: { throughput: 72, latencia: 32.4, erro: 0.028, ciclo: 6 } },
          { id: "P040102", name: "Solda", x: 253, y: 191, z: -123, metrics: { throughput: 154, latencia: 23.2, erro: 0.039, ciclo: 3 } },
          { id: "P040103", name: "Pintura", x: 212, y: 191, z: -125, metrics: { throughput: 125, latencia: 2.0, erro: 0.053, ciclo: 5 } },
          { id: "P040104", name: "Acabamento", x: 226, y: 187, z: -125, metrics: { throughput: 170, latencia: 4.3, erro: 0.05, ciclo: 6 } },
          { id: "P040105", name: "Expedição", x: 225, y: 189, z: -113, metrics: { throughput: 34, latencia: 8.8, erro: 0.057, ciclo: 3 } }
        ],
        workers: [
          { id: "W40101", nome: "Thiago Melo", cargo: "Gerente", salario: 7335, produtividade: 0.66, skills: ["Lideran\u00e7a", "Gest\u00e3o", "Planejamento"], vinculo: "Corte", horas_semanais: 44 },
          { id: "W40102", nome: "Tatiana Nunes", cargo: "Operador CNC", salario: 5377, produtividade: 0.6, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Solda", horas_semanais: 48 },
          { id: "W40103", nome: "Maria Oliveira", cargo: "Soldador", salario: 8494, produtividade: 0.75, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Pintura", horas_semanais: 44 },
          { id: "W40104", nome: "Gustavo Barros", cargo: "Pintor", salario: 5195, produtividade: 0.62, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Acabamento", horas_semanais: 44 },
          { id: "W40105", nome: "Bruno Cavalcanti", cargo: "Inspetor", salario: 7335, produtividade: 0.86, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Expedição", horas_semanais: 44 },
          { id: "W40106", nome: "Patrícia Rocha", cargo: "Logístico", salario: 4386, produtividade: 0.72, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Corte", horas_semanais: 40 },
          { id: "W40107", nome: "Pedro Almeida", cargo: "Engenheiro", salario: 4806, produtividade: 0.86, skills: ["Projeto", "C\u00e1lculo", "Gest\u00e3o"], vinculo: "Solda", horas_semanais: 40 }
        ]
      },
      {
        id: "E0402", name: "PlastiVida", x: 138, y: 174, z: -103, radius: 30, color: "#fca5a5",
        metrics: { receita: 1800, custo: 1500, funcionarios: 140, resiliencia: 0.7, sinergia: 0.6 },
        processos: [
          { id: "P040201", name: "Injeção", x: 141, y: 177, z: -108, metrics: { throughput: 97, latencia: 6.4, erro: 0.041, ciclo: 5 } },
          { id: "P040202", name: "Extrusão", x: 154, y: 171, z: -120, metrics: { throughput: 106, latencia: 16.9, erro: 0.058, ciclo: 4 } },
          { id: "P040203", name: "Reciclagem", x: 118, y: 191, z: -110, metrics: { throughput: 169, latencia: 38.5, erro: 0.058, ciclo: 1 } }
        ],
        workers: [
          { id: "W40201", nome: "Eduardo Freitas", cargo: "Operador", salario: 4722, produtividade: 0.69, skills: ["Opera\u00e7\u00e3o", "Controle", "Monitoramento"], vinculo: "Injeção", horas_semanais: 44 },
          { id: "W40202", nome: "Ricardo Lima", cargo: "Técnico", salario: 11350, produtividade: 0.7, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Extrusão", horas_semanais: 48 },
          { id: "W40203", nome: "Fernanda Costa", cargo: "Engenheiro", salario: 5031, produtividade: 0.92, skills: ["Projeto", "C\u00e1lculo", "Gest\u00e3o"], vinculo: "Reciclagem", horas_semanais: 48 },
          { id: "W40204", nome: "Rafael Correia", cargo: "Estoquista", salario: 4122, produtividade: 0.9, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Injeção", horas_semanais: 48 },
          { id: "W40205", nome: "Cláudia Rezende", cargo: "Supervisor", salario: 9186, produtividade: 0.95, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Extrusão", horas_semanais: 44 }
        ]
      },
      {
        id: "E0403", name: "Alimentos Sabor", x: 163, y: 218, z: -98, radius: 25, color: "#fecaca",
        metrics: { receita: 1200, custo: 900, funcionarios: 80, resiliencia: 0.75, sinergia: 0.55 },
        processos: [
          { id: "P040301", name: "Produção", x: 149, y: 200, z: -117, metrics: { throughput: 38, latencia: 1.4, erro: 0.048, ciclo: 2 } },
          { id: "P040302", name: "Embalagem", x: 175, y: 213, z: -117, metrics: { throughput: 49, latencia: 16.3, erro: 0.008, ciclo: 4 } },
          { id: "P040303", name: "Controle", x: 171, y: 200, z: -114, metrics: { throughput: 126, latencia: 27.2, erro: 0.022, ciclo: 5 } }
        ],
        workers: [
          { id: "W40301", nome: "Diego Cardoso", cargo: "Operador", salario: 5862, produtividade: 0.53, skills: ["Opera\u00e7\u00e3o", "Controle", "Monitoramento"], vinculo: "Produção", horas_semanais: 48 },
          { id: "W40302", nome: "Rafael Correia", cargo: "Auxiliar", salario: 5894, produtividade: 0.78, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Embalagem", horas_semanais: 40 },
          { id: "W40303", nome: "Patrícia Rocha", cargo: "Inspetor quality", salario: 7883, produtividade: 0.87, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Controle", horas_semanais: 48 },
          { id: "W40304", nome: "Adriana Monteiro", cargo: "Gerente", salario: 2546, produtividade: 0.63, skills: ["Lideran\u00e7a", "Gest\u00e3o", "Planejamento"], vinculo: "Produção", horas_semanais: 40 }
        ]
      }
    ]
  },
  {
    id: "S05", name: "Tecnologia", color: "#06b6d4", emoji: "💻",
    x: -180, y: 180, z: 80, radius: 68,
    metrics: { receita: 8500, custo: 5200, saude: 0.94, entropia: 0.06, funcionarios: 420 },
    empresas: [
      {
        id: "E0501", name: "CodeNexus", x: -140, y: 135, z: 119, radius: 40, color: "#22d3ee",
        metrics: { receita: 4000, custo: 2400, funcionarios: 180, resiliencia: 0.95, sinergia: 0.9 },
        processos: [
          { id: "P050101", name: "Backend", x: -158, y: 151, z: 134, metrics: { throughput: 189, latencia: 52.0, erro: 0.026, ciclo: 2 } },
          { id: "P050102", name: "Frontend", x: -139, y: 131, z: 128, metrics: { throughput: 107, latencia: 25.3, erro: 0.007, ciclo: 4 } },
          { id: "P050103", name: "DevOps", x: -145, y: 152, z: 115, metrics: { throughput: 100, latencia: 57.2, erro: 0.042, ciclo: 4 } },
          { id: "P050104", name: "DataScience", x: -161, y: 115, z: 104, metrics: { throughput: 28, latencia: 26.2, erro: 0.045, ciclo: 3 } },
          { id: "P050105", name: "Mobile", x: -157, y: 145, z: 102, metrics: { throughput: 155, latencia: 57.4, erro: 0.034, ciclo: 6 } },
          { id: "P050106", name: "QA", x: -158, y: 136, z: 121, metrics: { throughput: 175, latencia: 56.5, erro: 0.026, ciclo: 6 } }
        ],
        workers: [
          { id: "W50101", nome: "Fernanda Costa", cargo: "Dev Senior", salario: 7441, produtividade: 0.72, skills: ["Arquitetura", "Code Review", "Mentoria"], vinculo: "Backend", horas_semanais: 40 },
          { id: "W50102", nome: "Tatiana Nunes", cargo: "Dev Pleno", salario: 3496, produtividade: 0.98, skills: ["Desenvolvimento", "Testes", "Debug"], vinculo: "Frontend", horas_semanais: 44 },
          { id: "W50103", nome: "Mariana Silveira", cargo: "Dev Junior", salario: 9484, produtividade: 0.83, skills: ["Desenvolvimento", "Aprendizado"], vinculo: "DevOps", horas_semanais: 44 },
          { id: "W50104", nome: "Beatriz Dias", cargo: "Tech Lead", salario: 9767, produtividade: 0.54, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "DataScience", horas_semanais: 40 },
          { id: "W50105", nome: "Fábio Duarte", cargo: "DevOps", salario: 7777, produtividade: 0.79, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Mobile", horas_semanais: 40 },
          { id: "W50106", nome: "Maria Oliveira", cargo: "UI/UX", salario: 4567, produtividade: 0.63, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "QA", horas_semanais: 48 },
          { id: "W50107", nome: "Larissa Albuquerque", cargo: "Scrum Master", salario: 11484, produtividade: 0.84, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Backend", horas_semanais: 44 },
          { id: "W50108", nome: "Roberto Pinto", cargo: "PO", salario: 11192, produtividade: 0.64, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Frontend", horas_semanais: 48 }
        ]
      },
      {
        id: "E0502", name: "DataFlow AI", x: -244, y: 178, z: 33, radius: 30, color: "#67e8f9",
        metrics: { receita: 2800, custo: 1600, funcionarios: 100, resiliencia: 0.92, sinergia: 0.85 },
        processos: [
          { id: "P050201", name: "Treino ML", x: -258, y: 170, z: 32, metrics: { throughput: 91, latencia: 21.4, erro: 0.012, ciclo: 2 } },
          { id: "P050202", name: "Inferência", x: -233, y: 195, z: 38, metrics: { throughput: 22, latencia: 8.9, erro: 0.038, ciclo: 1 } },
          { id: "P050203", name: "MLOps", x: -264, y: 200, z: 46, metrics: { throughput: 60, latencia: 22.9, erro: 0.028, ciclo: 2 } }
        ],
        workers: [
          { id: "W50201", nome: "João Santos", cargo: "Data Scientist", salario: 4197, produtividade: 0.78, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Treino ML", horas_semanais: 40 },
          { id: "W50202", nome: "Isabela Castro", cargo: "ML Engineer", salario: 5035, produtividade: 0.82, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Inferência", horas_semanais: 40 },
          { id: "W50203", nome: "Rosângela Azevedo", cargo: "MLOps", salario: 4273, produtividade: 0.67, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "MLOps", horas_semanais: 48 },
          { id: "W50204", nome: "Thiago Melo", cargo: "Analista", salario: 8522, produtividade: 0.56, skills: ["An\u00e1lise", "SQL", "Relat\u00f3rios"], vinculo: "Treino ML", horas_semanais: 44 },
          { id: "W50205", nome: "Vanessa Lopes", cargo: "Engenheiro Dados", salario: 11906, produtividade: 0.61, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Inferência", horas_semanais: 44 }
        ]
      },
      {
        id: "E0503", name: "CloudNine", x: -230, y: 179, z: 86, radius: 22, color: "#a5f3fc",
        metrics: { receita: 1200, custo: 800, funcionarios: 45, resiliencia: 0.88, sinergia: 0.7 },
        processos: [
          { id: "P050301", name: "Infra", x: -213, y: 181, z: 97, metrics: { throughput: 160, latencia: 26.8, erro: 0.017, ciclo: 5 } },
          { id: "P050302", name: "Monitoramento", x: -248, y: 176, z: 93, metrics: { throughput: 33, latencia: 17.4, erro: 0.041, ciclo: 4 } }
        ],
        workers: [
          { id: "W50301", nome: "Bruno Cavalcanti", cargo: "SRE", salario: 7814, produtividade: 0.87, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Infra", horas_semanais: 48 },
          { id: "W50302", nome: "Thiago Melo", cargo: "Cloud Engineer", salario: 3891, produtividade: 0.92, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Monitoramento", horas_semanais: 40 },
          { id: "W50303", nome: "Carla Fonseca", cargo: "Analista TI", salario: 5079, produtividade: 0.86, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Infra", horas_semanais: 40 }
        ]
      }
    ]
  },
  {
    id: "S06", name: "Serviços", color: "#a855f7", emoji: "🔧",
    x: 100, y: -250, z: -60, radius: 60,
    metrics: { receita: 2800, custo: 2100, saude: 0.85, entropia: 0.14, funcionarios: 200 },
    empresas: [
      {
        id: "E0601", name: "FixIt Manutenção", x: 104, y: -237, z: -95, radius: 28, color: "#c084fc",
        metrics: { receita: 900, custo: 650, funcionarios: 60, resiliencia: 0.7, sinergia: 0.5 },
        processos: [
          { id: "P060101", name: "Residencial", x: 79, y: -231, z: -88, metrics: { throughput: 49, latencia: 8.2, erro: 0.032, ciclo: 2 } },
          { id: "P060102", name: "Comercial", x: 111, y: -227, z: -93, metrics: { throughput: 23, latencia: 24.1, erro: 0.045, ciclo: 4 } }
        ],
        workers: [
          { id: "W60101", nome: "Roberto Pinto", cargo: "Técnico", salario: 3244, produtividade: 0.61, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Residencial", horas_semanais: 48 },
          { id: "W60102", nome: "Larissa Albuquerque", cargo: "Eletricista", salario: 3397, produtividade: 0.5, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Comercial", horas_semanais: 44 },
          { id: "W60103", nome: "Thiago Melo", cargo: "Encanador", salario: 5960, produtividade: 0.87, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Residencial", horas_semanais: 44 }
        ]
      },
      {
        id: "E0602", name: "CleanPro Limpeza", x: 113, y: -277, z: -31, radius: 24, color: "#d8b4fe",
        metrics: { receita: 600, custo: 450, funcionarios: 50, resiliencia: 0.65, sinergia: 0.45 },
        processos: [
          { id: "P060201", name: "Diário", x: 117, y: -258, z: -28, metrics: { throughput: 27, latencia: 26.5, erro: 0.007, ciclo: 4 } },
          { id: "P060202", name: "Profunda", x: 125, y: -277, z: -18, metrics: { throughput: 25, latencia: 24.0, erro: 0.019, ciclo: 3 } }
        ],
        workers: [
          { id: "W60201", nome: "Carlos Souza", cargo: "Auxiliar", salario: 7629, produtividade: 0.78, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Diário", horas_semanais: 48 },
          { id: "W60202", nome: "André Carvalho", cargo: "Supervisor", salario: 9126, produtividade: 0.84, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Profunda", horas_semanais: 44 },
          { id: "W60203", nome: "Lucia Ferreira", cargo: "Administrador", salario: 7243, produtividade: 0.56, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Diário", horas_semanais: 40 }
        ]
      },
      {
        id: "E0603", name: "Advocacia Leal", x: 67, y: -260, z: -97, radius: 20, color: "#e9d5ff",
        metrics: { receita: 800, custo: 500, funcionarios: 25, resiliencia: 0.8, sinergia: 0.6 },
        processos: [
          { id: "P060301", name: "Cível", x: 51, y: -269, z: -105, metrics: { throughput: 49, latencia: 36.3, erro: 0.046, ciclo: 6 } },
          { id: "P060302", name: "Trabalhista", x: 46, y: -274, z: -77, metrics: { throughput: 131, latencia: 28.1, erro: 0.034, ciclo: 5 } },
          { id: "P060303", name: "Tributário", x: 70, y: -242, z: -81, metrics: { throughput: 169, latencia: 38.3, erro: 0.038, ciclo: 6 } }
        ],
        workers: [
          { id: "W60301", nome: "Beatriz Dias", cargo: "Advogado", salario: 5254, produtividade: 0.54, skills: ["Jurisprud\u00eancia", "Pe\u00e7as", "Audi\u00eancias"], vinculo: "Cível", horas_semanais: 48 },
          { id: "W60302", nome: "Diego Cardoso", cargo: "Estagiário", salario: 4368, produtividade: 0.75, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Trabalhista", horas_semanais: 40 },
          { id: "W60303", nome: "Paulo Mendonça", cargo: "Paralegal", salario: 8224, produtividade: 0.67, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Tributário", horas_semanais: 48 }
        ]
      }
    ]
  },
  {
    id: "S07", name: "Governo", color: "#3b82f6", emoji: "🏛️",
    x: 0, y: 100, z: -150, radius: 78,
    metrics: { receita: 12000, custo: 11500, saude: 0.72, entropia: 0.28, funcionarios: 850 },
    empresas: [
      {
        id: "E0701", name: "Prefeitura", x: -60, y: 145, z: -127, radius: 42, color: "#60a5fa",
        metrics: { receita: 6000, custo: 5800, funcionarios: 400, resiliencia: 0.7, sinergia: 0.5 },
        processos: [
          { id: "P070101", name: "Saúde Pub", x: -44, y: 132, z: -127, metrics: { throughput: 159, latencia: 28.8, erro: 0.01, ciclo: 1 } },
          { id: "P070102", name: "Educação Pub", x: -57, y: 126, z: -126, metrics: { throughput: 187, latencia: 5.5, erro: 0.039, ciclo: 1 } },
          { id: "P070103", name: "Saneamento", x: -70, y: 165, z: -119, metrics: { throughput: 117, latencia: 31.7, erro: 0.037, ciclo: 3 } },
          { id: "P070104", name: "Segurança", x: -62, y: 138, z: -123, metrics: { throughput: 109, latencia: 46.6, erro: 0.041, ciclo: 1 } },
          { id: "P070105", name: "Obras", x: -35, y: 160, z: -126, metrics: { throughput: 21, latencia: 20.1, erro: 0.034, ciclo: 4 } }
        ],
        workers: [
          { id: "W70101", nome: "Amanda Pereira", cargo: "Servidor", salario: 7461, produtividade: 0.88, skills: ["Legisla\u00e7\u00e3o", "Atendimento", "Processo"], vinculo: "Saúde Pub", horas_semanais: 48 },
          { id: "W70102", nome: "Marcos Ribeiro", cargo: "Diretor", salario: 3420, produtividade: 0.67, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Educação Pub", horas_semanais: 40 },
          { id: "W70103", nome: "Tatiana Nunes", cargo: "Analista", salario: 7585, produtividade: 0.72, skills: ["An\u00e1lise", "SQL", "Relat\u00f3rios"], vinculo: "Saneamento", horas_semanais: 40 },
          { id: "W70104", nome: "Lucia Ferreira", cargo: "Técnico", salario: 3432, produtividade: 0.68, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Segurança", horas_semanais: 44 },
          { id: "W70105", nome: "Renata Campos", cargo: "Assistente", salario: 3757, produtividade: 0.81, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Obras", horas_semanais: 40 },
          { id: "W70106", nome: "Henrique Viana", cargo: "Coordenador", salario: 10807, produtividade: 0.68, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Saúde Pub", horas_semanais: 48 }
        ]
      },
      {
        id: "E0702", name: "Detran", x: 64, y: 53, z: -118, radius: 28, color: "#93c5fd",
        metrics: { receita: 2000, custo: 2100, funcionarios: 200, resiliencia: 0.6, sinergia: 0.4 },
        processos: [
          { id: "P070201", name: "Licenças", x: 81, y: 55, z: -106, metrics: { throughput: 97, latencia: 1.6, erro: 0.019, ciclo: 2 } },
          { id: "P070202", name: "Multas", x: 60, y: 77, z: -107, metrics: { throughput: 54, latencia: 14.0, erro: 0.009, ciclo: 1 } },
          { id: "P070203", name: "CNH", x: 57, y: 78, z: -132, metrics: { throughput: 134, latencia: 46.4, erro: 0.05, ciclo: 5 } }
        ],
        workers: [
          { id: "W70201", nome: "Isabela Castro", cargo: "Servidor", salario: 4963, produtividade: 0.66, skills: ["Legisla\u00e7\u00e3o", "Atendimento", "Processo"], vinculo: "Licenças", horas_semanais: 48 },
          { id: "W70202", nome: "Letícia Martins", cargo: "Analista", salario: 4817, produtividade: 0.94, skills: ["An\u00e1lise", "SQL", "Relat\u00f3rios"], vinculo: "Multas", horas_semanais: 44 },
          { id: "W70203", nome: "Carla Fonseca", cargo: "Atendente", salario: 8921, produtividade: 0.56, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "CNH", horas_semanais: 48 },
          { id: "W70204", nome: "Larissa Albuquerque", cargo: "Fiscal", salario: 3888, produtividade: 0.65, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Licenças", horas_semanais: 44 },
          { id: "W70205", nome: "Rosângela Azevedo", cargo: "Gerente", salario: 7881, produtividade: 0.89, skills: ["Lideran\u00e7a", "Gest\u00e3o", "Planejamento"], vinculo: "Multas", horas_semanais: 48 }
        ]
      },
      {
        id: "E0703", name: "Casa Civil", x: -10, y: 143, z: -122, radius: 22, color: "#bfdbfe",
        metrics: { receita: 1500, custo: 1600, funcionarios: 150, resiliencia: 0.55, sinergia: 0.35 },
        processos: [
          { id: "P070301", name: "Protocolo", x: -17, y: 166, z: -114, metrics: { throughput: 64, latencia: 32.3, erro: 0.019, ciclo: 4 } },
          { id: "P070302", name: "Almoxarifado", x: -23, y: 141, z: -106, metrics: { throughput: 117, latencia: 28.0, erro: 0.018, ciclo: 4 } },
          { id: "P070303", name: "TI", x: -3, y: 151, z: -116, metrics: { throughput: 46, latencia: 49.1, erro: 0.048, ciclo: 2 } }
        ],
        workers: [
          { id: "W70301", nome: "Maria Oliveira", cargo: "Servidor", salario: 4647, produtividade: 0.79, skills: ["Legisla\u00e7\u00e3o", "Atendimento", "Processo"], vinculo: "Protocolo", horas_semanais: 40 },
          { id: "W70302", nome: "Larissa Albuquerque", cargo: "Assessor", salario: 5158, produtividade: 0.59, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Almoxarifado", horas_semanais: 48 },
          { id: "W70303", nome: "Diego Cardoso", cargo: "Secretário", salario: 5212, produtividade: 0.85, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "TI", horas_semanais: 40 },
          { id: "W70304", nome: "Paulo Mendonça", cargo: "Técnico", salario: 9231, produtividade: 0.67, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Protocolo", horas_semanais: 48 }
        ]
      }
    ]
  },
  {
    id: "S08", name: "Transporte", color: "#8b5cf6", emoji: "🚌",
    x: -150, y: -150, z: 100, radius: 55,
    metrics: { receita: 2200, custo: 1800, saude: 0.8, entropia: 0.16, funcionarios: 180 },
    empresas: [
      {
        id: "E0801", name: "ViaRápida", x: -151, y: -105, z: 115, radius: 28, color: "#a78bfa",
        metrics: { receita: 1000, custo: 800, funcionarios: 70, resiliencia: 0.75, sinergia: 0.55 },
        processos: [
          { id: "P080101", name: "Urbano", x: -167, y: -103, z: 100, metrics: { throughput: 61, latencia: 49.1, erro: 0.053, ciclo: 1 } },
          { id: "P080102", name: "Intermunicipal", x: -150, y: -127, z: 120, metrics: { throughput: 133, latencia: 22.8, erro: 0.024, ciclo: 1 } }
        ],
        workers: [
          { id: "W80101", nome: "Letícia Martins", cargo: "Motorista", salario: 10373, produtividade: 0.92, skills: ["Dire\u00e7\u00e3o", "Log\u00edstica", "Rotas"], vinculo: "Urbano", horas_semanais: 48 },
          { id: "W80102", nome: "João Santos", cargo: "Cobrador", salario: 4180, produtividade: 0.84, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Intermunicipal", horas_semanais: 48 },
          { id: "W80103", nome: "Sérgio Peixoto", cargo: "Mecânico", salario: 4542, produtividade: 0.64, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Urbano", horas_semanais: 40 }
        ]
      },
      {
        id: "E0802", name: "LogFast", x: -185, y: -100, z: 107, radius: 24, color: "#c4b5fd",
        metrics: { receita: 800, custo: 650, funcionarios: 50, resiliencia: 0.8, sinergia: 0.6 },
        processos: [
          { id: "P080201", name: "Last Mile", x: -171, y: -125, z: 92, metrics: { throughput: 9, latencia: 15.7, erro: 0.05, ciclo: 5 } },
          { id: "P080202", name: "Interestadual", x: -164, y: -87, z: 120, metrics: { throughput: 113, latencia: 7.1, erro: 0.018, ciclo: 3 } }
        ],
        workers: [
          { id: "W80201", nome: "Bruno Cavalcanti", cargo: "Motorista", salario: 7721, produtividade: 0.95, skills: ["Dire\u00e7\u00e3o", "Log\u00edstica", "Rotas"], vinculo: "Last Mile", horas_semanais: 48 },
          { id: "W80202", nome: "Beatriz Dias", cargo: "Operador", salario: 7993, produtividade: 0.88, skills: ["Opera\u00e7\u00e3o", "Controle", "Monitoramento"], vinculo: "Interestadual", horas_semanais: 40 },
          { id: "W80203", nome: "Carlos Souza", cargo: "Dispatcher", salario: 3127, produtividade: 0.64, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Last Mile", horas_semanais: 44 }
        ]
      },
      {
        id: "E0803", name: "Cargo Express", x: -159, y: -184, z: 87, radius: 20, color: "#ddd6fe",
        metrics: { receita: 500, custo: 380, funcionarios: 35, resiliencia: 0.7, sinergia: 0.5 },
        processos: [
          { id: "P080301", name: "Coleta", x: -157, y: -209, z: 106, metrics: { throughput: 95, latencia: 14.8, erro: 0.026, ciclo: 6 } },
          { id: "P080302", name: "Entrega", x: -142, y: -204, z: 100, metrics: { throughput: 97, latencia: 4.5, erro: 0.032, ciclo: 5 } }
        ],
        workers: [
          { id: "W80301", nome: "Ricardo Lima", cargo: "Motorista", salario: 9378, produtividade: 0.81, skills: ["Dire\u00e7\u00e3o", "Log\u00edstica", "Rotas"], vinculo: "Coleta", horas_semanais: 48 },
          { id: "W80302", nome: "João Santos", cargo: "Auxiliar", salario: 9989, produtividade: 0.53, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Entrega", horas_semanais: 44 },
          { id: "W80303", nome: "Rafael Correia", cargo: "Atendente", salario: 11282, produtividade: 0.51, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Coleta", horas_semanais: 48 }
        ]
      }
    ]
  },
  {
    id: "S09", name: "Energia", color: "#eab308", emoji: "⚡",
    x: 300, y: 100, z: 0, radius: 58,
    metrics: { receita: 3500, custo: 2800, saude: 0.86, entropia: 0.1, funcionarios: 160 },
    empresas: [
      {
        id: "E0901", name: "EletroSul", x: 256, y: 115, z: -33, radius: 32, color: "#facc15",
        metrics: { receita: 1800, custo: 1400, funcionarios: 80, resiliencia: 0.88, sinergia: 0.7 },
        processos: [
          { id: "P090101", name: "Geração", x: 233, y: 112, z: -19, metrics: { throughput: 91, latencia: 48.9, erro: 0.011, ciclo: 6 } },
          { id: "P090102", name: "Distribuição", x: 260, y: 134, z: -23, metrics: { throughput: 166, latencia: 11.3, erro: 0.009, ciclo: 6 } },
          { id: "P090103", name: "Manutenção", x: 280, y: 119, z: -51, metrics: { throughput: 80, latencia: 12.5, erro: 0.048, ciclo: 1 } }
        ],
        workers: [
          { id: "W90101", nome: "Mariana Silveira", cargo: "Engenheiro", salario: 10202, produtividade: 0.52, skills: ["Projeto", "C\u00e1lculo", "Gest\u00e3o"], vinculo: "Geração", horas_semanais: 44 },
          { id: "W90102", nome: "Lucas Batista", cargo: "Técnico", salario: 8616, produtividade: 0.62, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Distribuição", horas_semanais: 40 },
          { id: "W90103", nome: "Carlos Souza", cargo: "Operador", salario: 8350, produtividade: 0.88, skills: ["Opera\u00e7\u00e3o", "Controle", "Monitoramento"], vinculo: "Manutenção", horas_semanais: 44 },
          { id: "W90104", nome: "Priscila Ramos", cargo: "Inspetor", salario: 6450, produtividade: 0.85, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Geração", horas_semanais: 48 }
        ]
      },
      {
        id: "E0902", name: "SolarPrime", x: 314, y: 99, z: -35, radius: 26, color: "#fde047",
        metrics: { receita: 1200, custo: 900, funcionarios: 50, resiliencia: 0.9, sinergia: 0.75 },
        processos: [
          { id: "P090201", name: "Instalação", x: 312, y: 101, z: -30, metrics: { throughput: 195, latencia: 26.7, erro: 0.057, ciclo: 3 } },
          { id: "P090202", name: "Projeto", x: 300, y: 105, z: -24, metrics: { throughput: 99, latencia: 55.5, erro: 0.032, ciclo: 1 } },
          { id: "P090203", name: "Monitoramento", x: 335, y: 101, z: -50, metrics: { throughput: 115, latencia: 36.3, erro: 0.05, ciclo: 5 } }
        ],
        workers: [
          { id: "W90201", nome: "Amanda Pereira", cargo: "Engenheiro Solar", salario: 9024, produtividade: 0.97, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Instalação", horas_semanais: 48 },
          { id: "W90202", nome: "Thiago Melo", cargo: "Técnico", salario: 10256, produtividade: 0.62, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Projeto", horas_semanais: 48 },
          { id: "W90203", nome: "Mariana Silveira", cargo: "Projetista", salario: 5631, produtividade: 0.64, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Monitoramento", horas_semanais: 40 }
        ]
      },
      {
        id: "E0903", name: "BioFuel", x: 319, y: 144, z: -45, radius: 20, color: "#fef9c3",
        metrics: { receita: 600, custo: 450, funcionarios: 35, resiliencia: 0.7, sinergia: 0.5 },
        processos: [
          { id: "P090301", name: "Produção", x: 340, y: 141, z: -26, metrics: { throughput: 116, latencia: 16.8, erro: 0.059, ciclo: 1 } },
          { id: "P090302", name: "Distribuição", x: 298, y: 161, z: -25, metrics: { throughput: 108, latencia: 22.1, erro: 0.048, ciclo: 6 } }
        ],
        workers: [
          { id: "W90301", nome: "Isabela Castro", cargo: "Operador", salario: 3811, produtividade: 0.66, skills: ["Opera\u00e7\u00e3o", "Controle", "Monitoramento"], vinculo: "Produção", horas_semanais: 44 },
          { id: "W90302", nome: "Amanda Pereira", cargo: "Motorista", salario: 7521, produtividade: 0.71, skills: ["Dire\u00e7\u00e3o", "Log\u00edstica", "Rotas"], vinculo: "Distribuição", horas_semanais: 48 },
          { id: "W90303", nome: "Fernanda Costa", cargo: "Administrador", salario: 9481, produtividade: 0.58, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Produção", horas_semanais: 44 }
        ]
      }
    ]
  },
  {
    id: "S10", name: "Cultura", color: "#ec4899", emoji: "🎭",
    x: -100, y: 250, z: -80, radius: 50,
    metrics: { receita: 1500, custo: 1200, saude: 0.78, entropia: 0.2, funcionarios: 120 },
    empresas: [
      {
        id: "E1001", name: "StudioArte", x: -162, y: 219, z: -44, radius: 24, color: "#f472b6",
        metrics: { receita: 600, custo: 450, funcionarios: 40, resiliencia: 0.7, sinergia: 0.6 },
        processos: [
          { id: "P100101", name: "Design", x: -159, y: 217, z: -41, metrics: { throughput: 118, latencia: 45.9, erro: 0.035, ciclo: 5 } },
          { id: "P100102", name: "Produção", x: -164, y: 219, z: -44, metrics: { throughput: 171, latencia: 17.1, erro: 0.058, ciclo: 1 } },
          { id: "P100103", name: "Entrega", x: -140, y: 205, z: -33, metrics: { throughput: 137, latencia: 23.5, erro: 0.034, ciclo: 3 } }
        ],
        workers: [
          { id: "W100101", nome: "Juliana Mendes", cargo: "Designer", salario: 9679, produtividade: 0.52, skills: ["Criatividade", "Ferramentas", "UI"], vinculo: "Design", horas_semanais: 40 },
          { id: "W100102", nome: "Carlos Souza", cargo: "Artista", salario: 6366, produtividade: 0.87, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Produção", horas_semanais: 44 },
          { id: "W100103", nome: "Marcos Ribeiro", cargo: "Produtor", salario: 8433, produtividade: 0.68, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Entrega", horas_semanais: 48 }
        ]
      },
      {
        id: "E1002", name: "CineVida", x: -55, y: 224, z: -39, radius: 22, color: "#f9a8d4",
        metrics: { receita: 500, custo: 400, funcionarios: 35, resiliencia: 0.65, sinergia: 0.55 },
        processos: [
          { id: "P100201", name: "Produção", x: -52, y: 204, z: -39, metrics: { throughput: 175, latencia: 21.2, erro: 0.005, ciclo: 5 } },
          { id: "P100202", name: "Pós-produção", x: -62, y: 218, z: -49, metrics: { throughput: 187, latencia: 42.7, erro: 0.042, ciclo: 2 } },
          { id: "P100203", name: "Exibição", x: -30, y: 222, z: -27, metrics: { throughput: 62, latencia: 7.7, erro: 0.013, ciclo: 2 } }
        ],
        workers: [
          { id: "W100201", nome: "Paulo Mendonça", cargo: "Diretor", salario: 9811, produtividade: 0.6, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Produção", horas_semanais: 48 },
          { id: "W100202", nome: "Letícia Martins", cargo: "Cinegrafista", salario: 7178, produtividade: 0.83, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Pós-produção", horas_semanais: 44 },
          { id: "W100203", nome: "Mateus Lima", cargo: "Editor", salario: 5779, produtividade: 0.56, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Exibição", horas_semanais: 40 }
        ]
      },
      {
        id: "E1003", name: "Livraria Saber", x: -154, y: 259, z: -80, radius: 18, color: "#fbcfe8",
        metrics: { receita: 400, custo: 300, funcionarios: 25, resiliencia: 0.6, sinergia: 0.45 },
        processos: [
          { id: "P100301", name: "Vendas", x: -134, y: 280, z: -70, metrics: { throughput: 139, latencia: 24.9, erro: 0.025, ciclo: 5 } },
          { id: "P100302", name: "Eventos", x: -175, y: 242, z: -80, metrics: { throughput: 169, latencia: 4.9, erro: 0.028, ciclo: 5 } }
        ],
        workers: [
          { id: "W100301", nome: "Rafael Correia", cargo: "Vendedor", salario: 8412, produtividade: 0.77, skills: ["Comunica\u00e7\u00e3o", "Negocia\u00e7\u00e3o", "CRM"], vinculo: "Vendas", horas_semanais: 44 },
          { id: "W100302", nome: "Eduardo Freitas", cargo: "Eventos", salario: 10157, produtividade: 0.89, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Eventos", horas_semanais: 48 },
          { id: "W100303", nome: "Carlos Souza", cargo: "Administrador", salario: 4626, produtividade: 0.79, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Vendas", horas_semanais: 40 }
        ]
      }
    ]
  },
  {
    id: "S11", name: "Agro", color: "#22c55e", emoji: "🌾",
    x: 150, y: -180, z: -120, radius: 62,
    metrics: { receita: 3800, custo: 2900, saude: 0.84, entropia: 0.15, funcionarios: 250 },
    empresas: [
      {
        id: "E1101", name: "Fazenda Verde", x: 152, y: -230, z: -138, radius: 32, color: "#4ade80",
        metrics: { receita: 1800, custo: 1300, funcionarios: 120, resiliencia: 0.8, sinergia: 0.6 },
        processos: [
          { id: "P110101", name: "Plantio", x: 139, y: -215, z: -123, metrics: { throughput: 75, latencia: 7.9, erro: 0.019, ciclo: 5 } },
          { id: "P110102", name: "Colheita", x: 132, y: -223, z: -148, metrics: { throughput: 156, latencia: 56.5, erro: 0.01, ciclo: 6 } },
          { id: "P110103", name: "Armazenamento", x: 166, y: -209, z: -120, metrics: { throughput: 91, latencia: 50.6, erro: 0.034, ciclo: 1 } }
        ],
        workers: [
          { id: "W110101", nome: "Vanessa Lopes", cargo: "Agrônomo", salario: 5481, produtividade: 0.87, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Plantio", horas_semanais: 40 },
          { id: "W110102", nome: "Patrícia Rocha", cargo: "Tratorista", salario: 9588, produtividade: 0.74, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Colheita", horas_semanais: 40 },
          { id: "W110103", nome: "Paulo Mendonça", cargo: "Peão", salario: 4534, produtividade: 0.75, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Armazenamento", horas_semanais: 44 },
          { id: "W110104", nome: "Lucas Batista", cargo: "Gerente", salario: 5197, produtividade: 0.58, skills: ["Lideran\u00e7a", "Gest\u00e3o", "Planejamento"], vinculo: "Plantio", horas_semanais: 48 },
          { id: "W110105", nome: "Helena Moura", cargo: "Engenheiro Agr", salario: 6194, produtividade: 0.67, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Colheita", horas_semanais: 48 }
        ]
      },
      {
        id: "E1102", name: "AgroTech", x: 219, y: -176, z: -88, radius: 26, color: "#86efac",
        metrics: { receita: 1200, custo: 900, funcionarios: 60, resiliencia: 0.85, sinergia: 0.7 },
        processos: [
          { id: "P110201", name: "Sensoriamento", x: 213, y: -171, z: -93, metrics: { throughput: 180, latencia: 24.7, erro: 0.028, ciclo: 6 } },
          { id: "P110202", name: "Drone", x: 197, y: -191, z: -80, metrics: { throughput: 111, latencia: 29.3, erro: 0.013, ciclo: 5 } },
          { id: "P110203", name: "Irrigação", x: 203, y: -181, z: -88, metrics: { throughput: 192, latencia: 59.3, erro: 0.021, ciclo: 4 } }
        ],
        workers: [
          { id: "W110201", nome: "Pedro Almeida", cargo: "Engenheiro", salario: 11949, produtividade: 0.63, skills: ["Projeto", "C\u00e1lculo", "Gest\u00e3o"], vinculo: "Sensoriamento", horas_semanais: 40 },
          { id: "W110202", nome: "Maria Oliveira", cargo: "Técnico", salario: 11873, produtividade: 0.7, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Drone", horas_semanais: 48 },
          { id: "W110203", nome: "Sérgio Peixoto", cargo: "Piloto Drone", salario: 2996, produtividade: 0.74, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Irrigação", horas_semanais: 48 }
        ]
      },
      {
        id: "E1103", name: "Orgânicos Plus", x: 92, y: -176, z: -121, radius: 22, color: "#bbf7d0",
        metrics: { receita: 800, custo: 600, funcionarios: 40, resiliencia: 0.7, sinergia: 0.5 },
        processos: [
          { id: "P110301", name: "Produção", x: 106, y: -175, z: -126, metrics: { throughput: 45, latencia: 48.9, erro: 0.056, ciclo: 6 } },
          { id: "P110302", name: "Embalagem", x: 87, y: -189, z: -131, metrics: { throughput: 132, latencia: 59.8, erro: 0.029, ciclo: 3 } },
          { id: "P110303", name: "Venda", x: 98, y: -200, z: -136, metrics: { throughput: 105, latencia: 30.6, erro: 0.058, ciclo: 2 } }
        ],
        workers: [
          { id: "W110301", nome: "Patrícia Rocha", cargo: "Produtor", salario: 10937, produtividade: 0.77, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Produção", horas_semanais: 44 },
          { id: "W110302", nome: "Daniela Tavares", cargo: "Auxiliar", salario: 6460, produtividade: 0.72, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Embalagem", horas_semanais: 44 },
          { id: "W110303", nome: "Bruno Cavalcanti", cargo: "Vendedor", salario: 9866, produtividade: 0.62, skills: ["Comunica\u00e7\u00e3o", "Negocia\u00e7\u00e3o", "CRM"], vinculo: "Venda", horas_semanais: 40 }
        ]
      }
    ]
  },
  {
    id: "S12", name: "Finanças", color: "#14b8a6", emoji: "💰",
    x: -250, y: -100, z: -80, radius: 65,
    metrics: { receita: 4500, custo: 3200, saude: 0.9, entropia: 0.08, funcionarios: 200 },
    empresas: [
      {
        id: "E1201", name: "BancoSol", x: -247, y: -152, z: -81, radius: 35, color: "#2dd4bf",
        metrics: { receita: 2200, custo: 1500, funcionarios: 100, resiliencia: 0.92, sinergia: 0.8 },
        processos: [
          { id: "P120101", name: "Varejo", x: -240, y: -149, z: -66, metrics: { throughput: 76, latencia: 49.6, erro: 0.038, ciclo: 5 } },
          { id: "P120102", name: "Corporate", x: -265, y: -169, z: -95, metrics: { throughput: 105, latencia: 22.7, erro: 0.059, ciclo: 5 } },
          { id: "P120103", name: "Digital", x: -249, y: -129, z: -92, metrics: { throughput: 55, latencia: 36.3, erro: 0.025, ciclo: 1 } }
        ],
        workers: [
          { id: "W120101", nome: "Gustavo Barros", cargo: "Gerente", salario: 10210, produtividade: 0.64, skills: ["Lideran\u00e7a", "Gest\u00e3o", "Planejamento"], vinculo: "Varejo", horas_semanais: 40 },
          { id: "W120102", nome: "Vanessa Lopes", cargo: "Analista", salario: 4261, produtividade: 0.71, skills: ["An\u00e1lise", "SQL", "Relat\u00f3rios"], vinculo: "Corporate", horas_semanais: 44 },
          { id: "W120103", nome: "João Santos", cargo: "Caixa", salario: 8492, produtividade: 0.87, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Digital", horas_semanais: 44 },
          { id: "W120104", nome: "Isabela Castro", cargo: "Consultor", salario: 3241, produtividade: 0.69, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Varejo", horas_semanais: 48 },
          { id: "W120105", nome: "Eduardo Freitas", cargo: "TI", salario: 11704, produtividade: 0.59, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Corporate", horas_semanais: 48 }
        ]
      },
      {
        id: "E1202", name: "Seguros Vita", x: -239, y: -129, z: -80, radius: 28, color: "#5eead4",
        metrics: { receita: 1200, custo: 900, funcionarios: 50, resiliencia: 0.85, sinergia: 0.7 },
        processos: [
          { id: "P120201", name: "Auto", x: -225, y: -107, z: -81, metrics: { throughput: 156, latencia: 20.5, erro: 0.05, ciclo: 5 } },
          { id: "P120202", name: "Vida", x: -233, y: -109, z: -64, metrics: { throughput: 81, latencia: 28.8, erro: 0.002, ciclo: 3 } },
          { id: "P120203", name: "Empresarial", x: -221, y: -147, z: -74, metrics: { throughput: 154, latencia: 18.8, erro: 0.054, ciclo: 6 } }
        ],
        workers: [
          { id: "W120201", nome: "Maria Oliveira", cargo: "Corretor", salario: 7956, produtividade: 0.89, skills: ["Mercado", "Negocia\u00e7\u00e3o", "Capta\u00e7\u00e3o"], vinculo: "Auto", horas_semanais: 48 },
          { id: "W120202", nome: "Patrícia Rocha", cargo: "Analista", salario: 9987, produtividade: 0.57, skills: ["An\u00e1lise", "SQL", "Relat\u00f3rios"], vinculo: "Vida", horas_semanais: 48 },
          { id: "W120203", nome: "Mateus Lima", cargo: "Sinistro", salario: 4790, produtividade: 0.66, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Empresarial", horas_semanais: 48 }
        ]
      },
      {
        id: "E1203", name: "Invest Plus", x: -186, y: -73, z: -112, radius: 24, color: "#99f6e4",
        metrics: { receita: 1100, custo: 800, funcionarios: 35, resiliencia: 0.88, sinergia: 0.75 },
        processos: [
          { id: "P120301", name: "Renda Fixa", x: -168, y: -83, z: -130, metrics: { throughput: 151, latencia: 56.9, erro: 0.007, ciclo: 1 } },
          { id: "P120302", name: "Variável", x: -183, y: -78, z: -106, metrics: { throughput: 43, latencia: 25.1, erro: 0.013, ciclo: 5 } },
          { id: "P120303", name: "Consultoria", x: -162, y: -59, z: -102, metrics: { throughput: 193, latencia: 43.8, erro: 0.043, ciclo: 5 } }
        ],
        workers: [
          { id: "W120301", nome: "Helena Moura", cargo: "Analista", salario: 10275, produtividade: 0.63, skills: ["An\u00e1lise", "SQL", "Relat\u00f3rios"], vinculo: "Renda Fixa", horas_semanais: 48 },
          { id: "W120302", nome: "Carlos Souza", cargo: "Consultor", salario: 11977, produtividade: 0.78, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Variável", horas_semanais: 48 },
          { id: "W120303", nome: "Rosângela Azevedo", cargo: "Trader", salario: 3341, produtividade: 0.78, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Consultoria", horas_semanais: 40 }
        ]
      }
    ]
  },
  {
    id: "S13", name: "Habitação", color: "#f97316", emoji: "🏠",
    x: 100, y: 200, z: 120, radius: 52,
    metrics: { receita: 2500, custo: 2100, saude: 0.76, entropia: 0.19, funcionarios: 150 },
    empresas: [
      {
        id: "E1301", name: "ConstruMax", x: 93, y: 201, z: 141, radius: 28, color: "#fb923c",
        metrics: { receita: 1200, custo: 1000, funcionarios: 80, resiliencia: 0.7, sinergia: 0.5 },
        processos: [
          { id: "P130101", name: "Projeto", x: 87, y: 190, z: 140, metrics: { throughput: 78, latencia: 42.4, erro: 0.042, ciclo: 4 } },
          { id: "P130102", name: "Obra", x: 88, y: 206, z: 143, metrics: { throughput: 148, latencia: 56.1, erro: 0.048, ciclo: 3 } },
          { id: "P130103", name: "Acabamento", x: 86, y: 183, z: 157, metrics: { throughput: 178, latencia: 32.8, erro: 0.054, ciclo: 4 } }
        ],
        workers: [
          { id: "W130101", nome: "Francisco Gomes", cargo: "Engenheiro", salario: 10337, produtividade: 0.75, skills: ["Projeto", "C\u00e1lculo", "Gest\u00e3o"], vinculo: "Projeto", horas_semanais: 44 },
          { id: "W130102", nome: "Lucas Batista", cargo: "Pedreiro", salario: 5337, produtividade: 0.72, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Obra", horas_semanais: 48 },
          { id: "W130103", nome: "Amanda Pereira", cargo: "Mestre Obra", salario: 8108, produtividade: 0.76, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Acabamento", horas_semanais: 48 },
          { id: "W130104", nome: "Larissa Albuquerque", cargo: "Orçamentista", salario: 6831, produtividade: 0.79, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Projeto", horas_semanais: 40 }
        ]
      },
      {
        id: "E1302", name: "Imobiliária Sonho", x: 81, y: 267, z: 104, radius: 24, color: "#fdba74",
        metrics: { receita: 800, custo: 600, funcionarios: 40, resiliencia: 0.75, sinergia: 0.55 },
        processos: [
          { id: "P130201", name: "Vendas", x: 91, y: 286, z: 101, metrics: { throughput: 40, latencia: 7.0, erro: 0.045, ciclo: 2 } },
          { id: "P130202", name: "Administração", x: 71, y: 245, z: 117, metrics: { throughput: 62, latencia: 38.5, erro: 0.004, ciclo: 4 } }
        ],
        workers: [
          { id: "W130201", nome: "Vanessa Lopes", cargo: "Corretor", salario: 7258, produtividade: 0.52, skills: ["Mercado", "Negocia\u00e7\u00e3o", "Capta\u00e7\u00e3o"], vinculo: "Vendas", horas_semanais: 48 },
          { id: "W130202", nome: "Paulo Mendonça", cargo: "Gerente", salario: 3794, produtividade: 0.67, skills: ["Lideran\u00e7a", "Gest\u00e3o", "Planejamento"], vinculo: "Administração", horas_semanais: 44 },
          { id: "W130203", nome: "Marcos Ribeiro", cargo: "Administrativo", salario: 6701, produtividade: 0.86, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Vendas", horas_semanais: 40 }
        ]
      },
      {
        id: "E1303", name: "AlugueFácil", x: 152, y: 181, z: 166, radius: 20, color: "#fed7aa",
        metrics: { receita: 500, custo: 350, funcionarios: 25, resiliencia: 0.65, sinergia: 0.45 },
        processos: [
          { id: "P130301", name: "Locação", x: 145, y: 176, z: 164, metrics: { throughput: 170, latencia: 4.0, erro: 0.046, ciclo: 6 } },
          { id: "P130302", name: "Manutenção", x: 163, y: 170, z: 180, metrics: { throughput: 194, latencia: 43.5, erro: 0.056, ciclo: 2 } }
        ],
        workers: [
          { id: "W130301", nome: "Diego Cardoso", cargo: "Administrador", salario: 4147, produtividade: 0.83, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Locação", horas_semanais: 40 },
          { id: "W130302", nome: "Mateus Lima", cargo: "Técnico", salario: 2584, produtividade: 0.96, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Manutenção", horas_semanais: 40 },
          { id: "W130303", nome: "Renata Campos", cargo: "Atendente", salario: 9166, produtividade: 0.81, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Locação", horas_semanais: 44 }
        ]
      }
    ]
  },
  {
    id: "S14", name: "Saneamento", color: "#0ea5e9", emoji: "💧",
    x: -80, y: 80, z: 150, radius: 48,
    metrics: { receita: 1800, custo: 1500, saude: 0.82, entropia: 0.13, funcionarios: 120 },
    empresas: [
      {
        id: "E1401", name: "Água Clara", x: -65, y: 82, z: 158, radius: 26, color: "#38bdf8",
        metrics: { receita: 900, custo: 700, funcionarios: 60, resiliencia: 0.8, sinergia: 0.6 },
        processos: [
          { id: "P140101", name: "Tratamento", x: -49, y: 91, z: 171, metrics: { throughput: 131, latencia: 53.4, erro: 0.009, ciclo: 5 } },
          { id: "P140102", name: "Distribuição", x: -61, y: 74, z: 150, metrics: { throughput: 33, latencia: 20.2, erro: 0.044, ciclo: 6 } },
          { id: "P140103", name: "Coleta", x: -74, y: 102, z: 149, metrics: { throughput: 8, latencia: 44.3, erro: 0.048, ciclo: 5 } }
        ],
        workers: [
          { id: "W140101", nome: "Daniela Vieira", cargo: "Operador", salario: 9004, produtividade: 0.88, skills: ["Opera\u00e7\u00e3o", "Controle", "Monitoramento"], vinculo: "Tratamento", horas_semanais: 40 },
          { id: "W140102", nome: "Antônio Barbosa", cargo: "Técnico", salario: 7242, produtividade: 0.92, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Distribuição", horas_semanais: 40 },
          { id: "W140103", nome: "Maria Oliveira", cargo: "Engenheiro", salario: 7388, produtividade: 0.77, skills: ["Projeto", "C\u00e1lculo", "Gest\u00e3o"], vinculo: "Coleta", horas_semanais: 40 }
        ]
      },
      {
        id: "E1402", name: "EcoRecicla", x: -126, y: 57, z: 117, radius: 22, color: "#7dd3fc",
        metrics: { receita: 500, custo: 400, funcionarios: 35, resiliencia: 0.7, sinergia: 0.5 },
        processos: [
          { id: "P140201", name: "Coleta", x: -121, y: 52, z: 112, metrics: { throughput: 6, latencia: 16.0, erro: 0.015, ciclo: 3 } },
          { id: "P140202", name: "Triagem", x: -130, y: 51, z: 134, metrics: { throughput: 190, latencia: 34.6, erro: 0.016, ciclo: 3 } },
          { id: "P140203", name: "Reciclagem", x: -107, y: 47, z: 100, metrics: { throughput: 175, latencia: 7.5, erro: 0.019, ciclo: 4 } }
        ],
        workers: [
          { id: "W140201", nome: "Rodrigo Cunha", cargo: "Motorista", salario: 5375, produtividade: 0.79, skills: ["Dire\u00e7\u00e3o", "Log\u00edstica", "Rotas"], vinculo: "Coleta", horas_semanais: 48 },
          { id: "W140202", nome: "Daniela Tavares", cargo: "Triador", salario: 9143, produtividade: 0.9, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Triagem", horas_semanais: 48 },
          { id: "W140203", nome: "Camila Araújo", cargo: "Operador", salario: 7872, produtividade: 0.54, skills: ["Opera\u00e7\u00e3o", "Controle", "Monitoramento"], vinculo: "Reciclagem", horas_semanais: 48 }
        ]
      },
      {
        id: "E1403", name: "SaneaTech", x: -111, y: 126, z: 195, radius: 18, color: "#bae6fd",
        metrics: { receita: 400, custo: 300, funcionarios: 25, resiliencia: 0.75, sinergia: 0.55 },
        processos: [
          { id: "P140301", name: "Monitoramento", x: -98, y: 124, z: 201, metrics: { throughput: 184, latencia: 33.2, erro: 0.029, ciclo: 5 } },
          { id: "P140302", name: "Manutenção", x: -94, y: 114, z: 190, metrics: { throughput: 179, latencia: 45.4, erro: 0.053, ciclo: 1 } }
        ],
        workers: [
          { id: "W140301", nome: "Rodrigo Cunha", cargo: "Técnico", salario: 7595, produtividade: 0.83, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Monitoramento", horas_semanais: 48 },
          { id: "W140302", nome: "Mariana Silveira", cargo: "Analista", salario: 7333, produtividade: 0.68, skills: ["An\u00e1lise", "SQL", "Relat\u00f3rios"], vinculo: "Manutenção", horas_semanais: 40 },
          { id: "W140303", nome: "Mateus Lima", cargo: "Engenheiro", salario: 6090, produtividade: 0.96, skills: ["Projeto", "C\u00e1lculo", "Gest\u00e3o"], vinculo: "Monitoramento", horas_semanais: 44 }
        ]
      }
    ]
  },
  {
    id: "S15", name: "Segurança", color: "#dc2626", emoji: "🛡️",
    x: 50, y: -50, z: -200, radius: 56,
    metrics: { receita: 2000, custo: 1700, saude: 0.8, entropia: 0.17, funcionarios: 180 },
    empresas: [
      {
        id: "E1501", name: "GuardTech", x: 93, y: -91, z: -163, radius: 28, color: "#f87171",
        metrics: { receita: 1000, custo: 800, funcionarios: 90, resiliencia: 0.8, sinergia: 0.6 },
        processos: [
          { id: "P150101", name: "Monitoramento", x: 81, y: -71, z: -146, metrics: { throughput: 130, latencia: 5.9, erro: 0.031, ciclo: 1 } },
          { id: "P150102", name: "Ronda", x: 97, y: -108, z: -151, metrics: { throughput: 111, latencia: 27.7, erro: 0.004, ciclo: 4 } },
          { id: "P150103", name: "Central", x: 111, y: -97, z: -182, metrics: { throughput: 106, latencia: 15.6, erro: 0.001, ciclo: 2 } }
        ],
        workers: [
          { id: "W150101", nome: "Roberto Pinto", cargo: "Vigilante", salario: 3779, produtividade: 0.94, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Monitoramento", horas_semanais: 40 },
          { id: "W150102", nome: "Tatiana Nunes", cargo: "Operador CFTV", salario: 3511, produtividade: 0.9, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Ronda", horas_semanais: 48 },
          { id: "W150103", nome: "Mateus Lima", cargo: "Supervisor", salario: 5812, produtividade: 0.77, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Central", horas_semanais: 40 },
          { id: "W150104", nome: "Bruno Cavalcanti", cargo: "Gerente", salario: 5195, produtividade: 0.66, skills: ["Lideran\u00e7a", "Gest\u00e3o", "Planejamento"], vinculo: "Monitoramento", horas_semanais: 48 }
        ]
      },
      {
        id: "E1502", name: "CyberShield", x: 26, y: -86, z: -152, radius: 24, color: "#fca5a5",
        metrics: { receita: 600, custo: 450, funcionarios: 40, resiliencia: 0.85, sinergia: 0.7 },
        processos: [
          { id: "P150201", name: "Pentest", x: 42, y: -65, z: -146, metrics: { throughput: 100, latencia: 59.6, erro: 0.024, ciclo: 4 } },
          { id: "P150202", name: "SOC", x: 25, y: -106, z: -138, metrics: { throughput: 39, latencia: 39.4, erro: 0.022, ciclo: 2 } },
          { id: "P150203", name: "Compliance", x: 35, y: -86, z: -139, metrics: { throughput: 37, latencia: 43.8, erro: 0.014, ciclo: 1 } }
        ],
        workers: [
          { id: "W150201", nome: "Gustavo Barros", cargo: "Analista Security", salario: 9441, produtividade: 0.67, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Pentest", horas_semanais: 40 },
          { id: "W150202", nome: "Lucia Ferreira", cargo: "Pentester", salario: 11366, produtividade: 0.98, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "SOC", horas_semanais: 40 },
          { id: "W150203", nome: "Maria Oliveira", cargo: "Engenheiro", salario: 10234, produtividade: 0.52, skills: ["Projeto", "C\u00e1lculo", "Gest\u00e3o"], vinculo: "Compliance", horas_semanais: 44 }
        ]
      },
      {
        id: "E1503", name: "SafeHome", x: 50, y: -72, z: -158, radius: 20, color: "#fecaca",
        metrics: { receita: 400, custo: 300, funcionarios: 30, resiliencia: 0.7, sinergia: 0.5 },
        processos: [
          { id: "P150301", name: "Instalação", x: 73, y: -90, z: -176, metrics: { throughput: 174, latencia: 25.4, erro: 0.046, ciclo: 1 } },
          { id: "P150302", name: "Monitoramento", x: 40, y: -84, z: -174, metrics: { throughput: 30, latencia: 35.8, erro: 0.027, ciclo: 6 } }
        ],
        workers: [
          { id: "W150301", nome: "Daniela Tavares", cargo: "Técnico", salario: 10085, produtividade: 0.82, skills: ["Manuten\u00e7\u00e3o", "Diagn\u00f3stico", "Reparo"], vinculo: "Instalação", horas_semanais: 48 },
          { id: "W150302", nome: "Carlos Souza", cargo: "Instalador", salario: 9448, produtividade: 0.76, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Monitoramento", horas_semanais: 40 },
          { id: "W150303", nome: "Thiago Melo", cargo: "Atendente", salario: 6555, produtividade: 0.72, skills: ["Comunica\u00e7\u00e3o", "Organiza\u00e7\u00e3o", "Responsabilidade"], vinculo: "Instalação", horas_semanais: 40 }
        ]
      }
    ]
  }
];

const CITY_EDGES = [
  { from: "S01", to: "S02", type: "funding", weight: 0.65 },
  { from: "S01", to: "S03", type: "funding", weight: 0.72 },
  { from: "S01", to: "S05", type: "recursos", weight: 0.72 },
  { from: "S01", to: "S06", type: "recursos", weight: 0.72 },
  { from: "S01", to: "S11", type: "funding", weight: 0.55 },
  { from: "S01", to: "S12", type: "recursos", weight: 0.59 },
  { from: "S01", to: "S13", type: "recursos", weight: 0.65 },
  { from: "S01", to: "S15", type: "dados", weight: 0.51 },
  { from: "S02", to: "S03", type: "funding", weight: 0.77 },
  { from: "S02", to: "S04", type: "funding", weight: 0.73 },
  { from: "S02", to: "S05", type: "dados", weight: 0.69 },
  { from: "S02", to: "S08", type: "funding", weight: 0.73 },
  { from: "S02", to: "S11", type: "recursos", weight: 0.79 },
  { from: "S02", to: "S14", type: "funding", weight: 0.52 },
  { from: "S02", to: "S15", type: "dados", weight: 0.88 },
  { from: "S03", to: "S06", type: "dados", weight: 0.64 },
  { from: "S03", to: "S07", type: "recursos", weight: 0.6 },
  { from: "S03", to: "S09", type: "dados", weight: 0.66 },
  { from: "S03", to: "S10", type: "funding", weight: 0.89 },
  { from: "S03", to: "S11", type: "dados", weight: 0.81 },
  { from: "S03", to: "S13", type: "dados", weight: 0.64 },
  { from: "S04", to: "S05", type: "recursos", weight: 0.67 },
  { from: "S04", to: "S06", type: "recursos", weight: 0.86 },
  { from: "S04", to: "S08", type: "recursos", weight: 0.64 },
  { from: "S04", to: "S09", type: "funding", weight: 0.71 },
  { from: "S04", to: "S10", type: "dados", weight: 0.52 },
  { from: "S04", to: "S11", type: "recursos", weight: 0.89 },
  { from: "S04", to: "S12", type: "funding", weight: 0.68 },
  { from: "S04", to: "S13", type: "funding", weight: 0.68 },
  { from: "S05", to: "S09", type: "recursos", weight: 0.86 },
  { from: "S05", to: "S11", type: "dados", weight: 0.66 },
  { from: "S05", to: "S12", type: "recursos", weight: 0.82 },
  { from: "S05", to: "S13", type: "dados", weight: 0.52 },
  { from: "S05", to: "S15", type: "recursos", weight: 0.72 },
  { from: "S06", to: "S08", type: "recursos", weight: 0.61 },
  { from: "S06", to: "S11", type: "recursos", weight: 0.9 },
  { from: "S06", to: "S12", type: "funding", weight: 0.55 },
  { from: "S06", to: "S13", type: "funding", weight: 0.8 },
  { from: "S06", to: "S14", type: "recursos", weight: 0.64 },
  { from: "S07", to: "S10", type: "recursos", weight: 0.9 },
  { from: "S07", to: "S12", type: "funding", weight: 0.59 },
  { from: "S07", to: "S13", type: "funding", weight: 0.68 },
  { from: "S07", to: "S14", type: "recursos", weight: 0.68 },
  { from: "S07", to: "S15", type: "recursos", weight: 0.73 },
  { from: "S08", to: "S13", type: "recursos", weight: 0.71 },
  { from: "S08", to: "S15", type: "funding", weight: 0.55 },
  { from: "S09", to: "S10", type: "funding", weight: 0.56 },
  { from: "S09", to: "S11", type: "dados", weight: 0.81 },
  { from: "S09", to: "S12", type: "recursos", weight: 0.79 },
  { from: "S09", to: "S13", type: "funding", weight: 0.63 },
  { from: "S10", to: "S11", type: "recursos", weight: 0.87 },
  { from: "S10", to: "S12", type: "recursos", weight: 0.84 },
  { from: "S10", to: "S15", type: "dados", weight: 0.85 },
  { from: "S11", to: "S14", type: "funding", weight: 0.6 },
  { from: "S12", to: "S13", type: "funding", weight: 0.54 },
  { from: "S12", to: "S14", type: "recursos", weight: 0.88 },
  { from: "S12", to: "S15", type: "funding", weight: 0.74 },
  { from: "S13", to: "S14", type: "recursos", weight: 0.66 },
  { from: "S14", to: "S15", type: "recursos", weight: 0.57 },
  { from: "E0102", to: "E0201", type: "dados", weight: 0.75 },
  { from: "E0103", to: "E0303", type: "recursos", weight: 0.41 },
  { from: "E0102", to: "E0403", type: "dados", weight: 0.44 },
  { from: "E0101", to: "E0501", type: "dados", weight: 0.37 },
  { from: "E0103", to: "E0601", type: "funding", weight: 0.15 },
  { from: "E0101", to: "E0701", type: "recursos", weight: 0.74 },
  { from: "E0101", to: "E0803", type: "dependencia", weight: 0.3 },
  { from: "E0101", to: "E0903", type: "dependencia", weight: 0.24 },
  { from: "E0101", to: "E1002", type: "dados", weight: 0.67 },
  { from: "E0101", to: "E1101", type: "recursos", weight: 0.58 },
  { from: "E0102", to: "E1202", type: "funding", weight: 0.45 },
  { from: "E0103", to: "E1303", type: "funding", weight: 0.79 },
  { from: "E0101", to: "E1402", type: "funding", weight: 0.62 },
  { from: "E0101", to: "E1501", type: "dependencia", weight: 0.27 },
  { from: "E0201", to: "E0304", type: "dependencia", weight: 0.27 },
  { from: "E0201", to: "E0401", type: "funding", weight: 0.44 },
  { from: "E0201", to: "E0503", type: "funding", weight: 0.8 },
  { from: "E0201", to: "E0601", type: "dependencia", weight: 0.26 },
  { from: "E0202", to: "E0701", type: "funding", weight: 0.71 },
  { from: "E0201", to: "E0802", type: "dependencia", weight: 0.15 },
  { from: "E0202", to: "E0902", type: "dependencia", weight: 0.34 },
  { from: "E0201", to: "E1001", type: "funding", weight: 0.34 },
  { from: "E0203", to: "E1103", type: "recursos", weight: 0.5 },
  { from: "E0201", to: "E1201", type: "dados", weight: 0.53 },
  { from: "E0201", to: "E1303", type: "dados", weight: 0.2 },
  { from: "E0201", to: "E1403", type: "dados", weight: 0.19 },
  { from: "E0202", to: "E1501", type: "dados", weight: 0.76 },
  { from: "E0301", to: "E0403", type: "recursos", weight: 0.2 },
  { from: "E0303", to: "E0501", type: "recursos", weight: 0.59 },
  { from: "E0302", to: "E0602", type: "recursos", weight: 0.41 },
  { from: "E0304", to: "E0702", type: "funding", weight: 0.75 },
  { from: "E0301", to: "E0802", type: "funding", weight: 0.55 },
  { from: "E0301", to: "E0903", type: "recursos", weight: 0.29 },
  { from: "E0302", to: "E1002", type: "dados", weight: 0.72 },
  { from: "E0303", to: "E1102", type: "recursos", weight: 0.25 },
  { from: "E0301", to: "E1203", type: "recursos", weight: 0.68 },
  { from: "E0301", to: "E1301", type: "funding", weight: 0.32 },
  { from: "E0302", to: "E1402", type: "dados", weight: 0.33 },
  { from: "E0303", to: "E1501", type: "recursos", weight: 0.6 },
  { from: "E0403", to: "E0502", type: "funding", weight: 0.31 },
  { from: "E0401", to: "E0603", type: "recursos", weight: 0.43 },
  { from: "E0401", to: "E0701", type: "dados", weight: 0.37 },
  { from: "E0403", to: "E0802", type: "dependencia", weight: 0.26 },
  { from: "E0403", to: "E0903", type: "funding", weight: 0.43 },
  { from: "E0401", to: "E1001", type: "dados", weight: 0.76 },
  { from: "E0403", to: "E1101", type: "dados", weight: 0.69 },
  { from: "E0402", to: "E1201", type: "recursos", weight: 0.18 },
  { from: "E0401", to: "E1302", type: "dados", weight: 0.29 },
  { from: "E0403", to: "E1401", type: "dependencia", weight: 0.38 },
  { from: "E0403", to: "E1503", type: "dados", weight: 0.25 },
  { from: "E0501", to: "E0601", type: "funding", weight: 0.72 },
  { from: "E0501", to: "E0703", type: "dependencia", weight: 0.75 },
  { from: "E0503", to: "E0803", type: "recursos", weight: 0.68 },
  { from: "E0501", to: "E0903", type: "funding", weight: 0.22 },
  { from: "E0502", to: "E1001", type: "dependencia", weight: 0.28 },
  { from: "E0502", to: "E1102", type: "dados", weight: 0.68 },
  { from: "E0501", to: "E1201", type: "dependencia", weight: 0.58 },
  { from: "E0502", to: "E1302", type: "recursos", weight: 0.71 },
  { from: "E0502", to: "E1403", type: "recursos", weight: 0.48 },
  { from: "E0501", to: "E1501", type: "recursos", weight: 0.72 },
  { from: "E0601", to: "E0703", type: "recursos", weight: 0.78 },
  { from: "E0601", to: "E0801", type: "recursos", weight: 0.54 },
  { from: "E0603", to: "E0902", type: "funding", weight: 0.43 },
  { from: "E0602", to: "E1003", type: "funding", weight: 0.27 },
  { from: "E0603", to: "E1102", type: "dados", weight: 0.15 },
  { from: "E0602", to: "E1202", type: "recursos", weight: 0.2 },
  { from: "E0603", to: "E1302", type: "funding", weight: 0.58 },
  { from: "E0603", to: "E1402", type: "dependencia", weight: 0.48 },
  { from: "E0602", to: "E1502", type: "dados", weight: 0.6 },
  { from: "E0701", to: "E0802", type: "dependencia", weight: 0.58 },
  { from: "E0703", to: "E0903", type: "recursos", weight: 0.74 },
  { from: "E0702", to: "E1003", type: "recursos", weight: 0.69 },
  { from: "E0702", to: "E1101", type: "recursos", weight: 0.43 },
  { from: "E0702", to: "E1202", type: "dados", weight: 0.44 },
  { from: "E0703", to: "E1302", type: "dados", weight: 0.67 },
  { from: "E0703", to: "E1401", type: "recursos", weight: 0.33 },
  { from: "E0701", to: "E1501", type: "dados", weight: 0.59 },
  { from: "E0801", to: "E0901", type: "dados", weight: 0.17 },
  { from: "E0802", to: "E1003", type: "funding", weight: 0.7 },
  { from: "E0802", to: "E1102", type: "dados", weight: 0.73 },
  { from: "E0803", to: "E1203", type: "dependencia", weight: 0.4 },
  { from: "E0801", to: "E1301", type: "funding", weight: 0.58 },
  { from: "E0801", to: "E1402", type: "dependencia", weight: 0.29 },
  { from: "E0801", to: "E1503", type: "funding", weight: 0.31 },
  { from: "E0902", to: "E1001", type: "recursos", weight: 0.67 },
  { from: "E0902", to: "E1103", type: "dependencia", weight: 0.68 },
  { from: "E0903", to: "E1202", type: "dependencia", weight: 0.73 },
  { from: "E0902", to: "E1302", type: "recursos", weight: 0.64 },
  { from: "E0903", to: "E1402", type: "recursos", weight: 0.56 },
  { from: "E0902", to: "E1501", type: "dados", weight: 0.61 },
  { from: "E1002", to: "E1102", type: "funding", weight: 0.36 },
  { from: "E1001", to: "E1201", type: "dados", weight: 0.3 },
  { from: "E1003", to: "E1301", type: "dependencia", weight: 0.19 },
  { from: "E1002", to: "E1403", type: "funding", weight: 0.45 },
  { from: "E1001", to: "E1502", type: "funding", weight: 0.4 },
  { from: "E1103", to: "E1202", type: "recursos", weight: 0.46 },
  { from: "E1102", to: "E1302", type: "dependencia", weight: 0.7 },
  { from: "E1103", to: "E1403", type: "dados", weight: 0.63 },
  { from: "E1102", to: "E1501", type: "dependencia", weight: 0.33 },
  { from: "E1201", to: "E1302", type: "dependencia", weight: 0.37 },
  { from: "E1203", to: "E1401", type: "dados", weight: 0.7 },
  { from: "E1203", to: "E1503", type: "dependencia", weight: 0.17 },
  { from: "E1303", to: "E1403", type: "funding", weight: 0.58 },
  { from: "E1303", to: "E1502", type: "dependencia", weight: 0.24 },
  { from: "E1401", to: "E1501", type: "recursos", weight: 0.32 },
  { from: "E0101", to: "E0102", type: "dados", weight: 0.42 },
  { from: "E0101", to: "E0103", type: "dados", weight: 0.49 },
  { from: "E0102", to: "E0103", type: "dados", weight: 0.45 },
  { from: "E0201", to: "E0202", type: "recursos", weight: 0.44 },
  { from: "E0201", to: "E0203", type: "recursos", weight: 0.56 },
  { from: "E0202", to: "E0203", type: "dados", weight: 0.85 },
  { from: "E0301", to: "E0302", type: "recursos", weight: 0.42 },
  { from: "E0301", to: "E0303", type: "recursos", weight: 0.73 },
  { from: "E0301", to: "E0304", type: "recursos", weight: 0.71 },
  { from: "E0302", to: "E0303", type: "recursos", weight: 0.41 },
  { from: "E0302", to: "E0304", type: "recursos", weight: 0.64 },
  { from: "E0303", to: "E0304", type: "recursos", weight: 0.37 },
  { from: "E0401", to: "E0402", type: "recursos", weight: 0.61 },
  { from: "E0401", to: "E0403", type: "dados", weight: 0.56 },
  { from: "E0402", to: "E0403", type: "recursos", weight: 0.72 },
  { from: "E0501", to: "E0502", type: "recursos", weight: 0.63 },
  { from: "E0501", to: "E0503", type: "recursos", weight: 0.73 },
  { from: "E0502", to: "E0503", type: "recursos", weight: 0.83 },
  { from: "E0601", to: "E0602", type: "dados", weight: 0.62 },
  { from: "E0601", to: "E0603", type: "dados", weight: 0.62 },
  { from: "E0602", to: "E0603", type: "recursos", weight: 0.52 },
  { from: "E0701", to: "E0702", type: "recursos", weight: 0.34 },
  { from: "E0701", to: "E0703", type: "dados", weight: 0.36 },
  { from: "E0702", to: "E0703", type: "recursos", weight: 0.77 },
  { from: "E0801", to: "E0802", type: "recursos", weight: 0.74 },
  { from: "E0801", to: "E0803", type: "dados", weight: 0.49 },
  { from: "E0802", to: "E0803", type: "recursos", weight: 0.66 },
  { from: "E0901", to: "E0902", type: "dados", weight: 0.58 },
  { from: "E0901", to: "E0903", type: "dados", weight: 0.31 },
  { from: "E0902", to: "E0903", type: "recursos", weight: 0.75 },
  { from: "E1001", to: "E1002", type: "dados", weight: 0.54 },
  { from: "E1001", to: "E1003", type: "dados", weight: 0.65 },
  { from: "E1002", to: "E1003", type: "dados", weight: 0.69 },
  { from: "E1101", to: "E1102", type: "dados", weight: 0.4 },
  { from: "E1101", to: "E1103", type: "dados", weight: 0.64 },
  { from: "E1102", to: "E1103", type: "dados", weight: 0.56 },
  { from: "E1201", to: "E1202", type: "recursos", weight: 0.48 },
  { from: "E1201", to: "E1203", type: "recursos", weight: 0.76 },
  { from: "E1202", to: "E1203", type: "dados", weight: 0.45 },
  { from: "E1301", to: "E1302", type: "dados", weight: 0.43 },
  { from: "E1301", to: "E1303", type: "dados", weight: 0.61 },
  { from: "E1302", to: "E1303", type: "dados", weight: 0.49 },
  { from: "E1401", to: "E1402", type: "recursos", weight: 0.77 },
  { from: "E1401", to: "E1403", type: "dados", weight: 0.57 },
  { from: "E1402", to: "E1403", type: "dados", weight: 0.74 },
  { from: "E1501", to: "E1502", type: "dados", weight: 0.45 },
  { from: "E1501", to: "E1503", type: "recursos", weight: 0.61 },
  { from: "E1502", to: "E1503", type: "dados", weight: 0.35 }
];

// ═══════════════════════════════════════════════════════════════
// INFRAESTRUTURA DE TRANSPORTE — Mapa da Cidade Tensorial
// 135 rotas terrestres | 10 rotas aéreas | 46 pontos de parada
// ═══════════════════════════════════════════════════════════════

const CITY_ROUTES = [
  { type:"rodovia", name:"Rodovia Central", from:{x:0,y:-200,z:0}, to:{x:0,y:100,z:-150}, mid:{x:0.0,y:-29.0,z:-75.0}, width:3.5, color:"#fbbf24", speed:120, capacity:"alta" },
  { type:"rodovia", name:"Via Industrial", from:{x:250,y:-80,z:60}, to:{x:200,y:150,z:-80}, mid:{x:225.0,y:21.0,z:-10.0}, width:3.5, color:"#fbbf24", speed:120, capacity:"alta" },
  { type:"rodovia", name:"Corredor Tech", from:{x:-180,y:180,z:80}, to:{x:-280,y:0,z:-50}, mid:{x:-230.0,y:80.0,z:15.0}, width:3.5, color:"#fbbf24", speed:120, capacity:"alta" },
  { type:"rodovia", name:"Via Financeira", from:{x:-250,y:-100,z:-80}, to:{x:250,y:-80,z:60}, mid:{x:0.0,y:-108.0,z:-10.0}, width:3.5, color:"#fbbf24", speed:120, capacity:"alta" },
  { type:"rodovia", name:"Estrada Energia", from:{x:300,y:100,z:0}, to:{x:200,y:150,z:-80}, mid:{x:250.0,y:110.0,z:-40.0}, width:3.5, color:"#fbbf24", speed:120, capacity:"alta" },
  { type:"rodovia", name:"Via Segurança", from:{x:0,y:100,z:-150}, to:{x:50,y:-50,z:-200}, mid:{x:25.0,y:7.0,z:-175.0}, width:3.5, color:"#fbbf24", speed:120, capacity:"alta" },
  { type:"rodovia", name:"Rota Agro-Saúde", from:{x:0,y:-200,z:0}, to:{x:150,y:-180,z:-120}, mid:{x:75.0,y:-168.0,z:-60.0}, width:3.5, color:"#fbbf24", speed:120, capacity:"alta" },
  { type:"rodovia", name:"Via Logística", from:{x:-150,y:-150,z:100}, to:{x:250,y:-80,z:60}, mid:{x:50.0,y:-85.0,z:80.0}, width:3.5, color:"#fbbf24", speed:120, capacity:"alta" },
  { type:"rodovia", name:"Corredor Digital", from:{x:-180,y:180,z:80}, to:{x:-250,y:-100,z:-80}, mid:{x:-215.0,y:17.0,z:0.0}, width:3.5, color:"#fbbf24", speed:120, capacity:"alta" },
  { type:"rodovia", name:"Via Saneamento", from:{x:-80,y:80,z:150}, to:{x:0,y:100,z:-150}, mid:{x:-40.0,y:78.0,z:0.0}, width:3.5, color:"#fbbf24", speed:120, capacity:"alta" },
  { type:"estrada", name:"Estrada Saúde-Serviços", from:{x:0,y:-200,z:0}, to:{x:100,y:-250,z:-60}, mid:{x:60.0,y:-225.0,z:-15.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Via Rural", from:{x:250,y:-80,z:60}, to:{x:150,y:-180,z:-120}, mid:{x:219.0,y:-130.0,z:-35.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Estrada Industrial", from:{x:200,y:150,z:-80}, to:{x:300,y:100,z:0}, mid:{x:239.0,y:125.0,z:-25.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Corredor Criativo", from:{x:-180,y:180,z:80}, to:{x:-100,y:250,z:-80}, mid:{x:-120.0,y:215.0,z:12.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Via Academia", from:{x:-280,y:0,z:-50}, to:{x:-250,y:-100,z:-80}, mid:{x:-285.0,y:-50.0,z:-68.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Estrada Habitacional", from:{x:100,y:200,z:120}, to:{x:-80,y:80,z:150}, mid:{x:-9.0,y:140.0,z:146.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Via Patrulha", from:{x:50,y:-50,z:-200}, to:{x:-150,y:-150,z:100}, mid:{x:-59.0,y:-100.0,z:-58.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Estrada Serviços", from:{x:100,y:-250,z:-60}, to:{x:-250,y:-100,z:-80}, mid:{x:-63.0,y:-175.0,z:-66.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Via Cultural", from:{x:-100,y:250,z:-80}, to:{x:100,y:200,z:120}, mid:{x:-9.0,y:225.0,z:21.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Rota Entrega", from:{x:150,y:-180,z:-120}, to:{x:-150,y:-150,z:100}, mid:{x:0.0,y:-165.0,z:-23.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Via Pública", from:{x:0,y:100,z:-150}, to:{x:-80,y:80,z:150}, mid:{x:-55.0,y:90.0,z:15.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"estrada", name:"Estrada Emergência", from:{x:0,y:-200,z:0}, to:{x:-250,y:-100,z:-80}, mid:{x:-129.0,y:-150.0,z:-26.0}, width:2.0, color:"#94a3b8", speed:80, capacity:"média" },
  { type:"avenida", name:"Av. Saúde 1", from:{x:0,y:-200,z:0}, to:{x:98,y:-152,z:12}, mid:{x:49,y:-176,z:6}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S01" },
  { type:"avenida", name:"Av. Saúde 2", from:{x:0,y:-200,z:0}, to:{x:-206,y:-278,z:31}, mid:{x:-103,y:-239,z:15}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S01" },
  { type:"avenida", name:"Av. Educação 1", from:{x:-280,y:0,z:-50}, to:{x:-193,y:35,z:-72}, mid:{x:-237,y:17,z:-61}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S02" },
  { type:"avenida", name:"Av. Comércio 1", from:{x:250,y:-80,z:60}, to:{x:468,y:-15,z:24}, mid:{x:359,y:-48,z:42}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S03" },
  { type:"avenida", name:"Av. Indústria 1", from:{x:200,y:150,z:-80}, to:{x:376,y:214,z:-78}, mid:{x:288,y:182,z:-79}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S04" },
  { type:"avenida", name:"Av. Tecnologia 1", from:{x:-180,y:180,z:80}, to:{x:-5,y:239,z:71}, mid:{x:-93,y:209,z:75}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S05" },
  { type:"avenida", name:"Av. Serviços 1", from:{x:100,y:-250,z:-60}, to:{x:247,y:-191,z:-14}, mid:{x:173,y:-221,z:-37}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S06" },
  { type:"avenida", name:"Av. Governo 1", from:{x:0,y:100,z:-150}, to:{x:207,y:163,z:-188}, mid:{x:103,y:131,z:-169}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S07" },
  { type:"avenida", name:"Av. Transporte 1", from:{x:-150,y:-150,z:100}, to:{x:-60,y:-113,z:74}, mid:{x:-105,y:-132,z:87}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S08" },
  { type:"avenida", name:"Av. Transporte 2", from:{x:-150,y:-150,z:100}, to:{x:-331,y:-240,z:105}, mid:{x:-241,y:-195,z:102}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S08" },
  { type:"avenida", name:"Av. Energia 1", from:{x:300,y:100,z:0}, to:{x:432,y:144,z:-27}, mid:{x:366,y:122,z:-14}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S09" },
  { type:"avenida", name:"Av. Cultura 1", from:{x:-100,y:250,z:-80}, to:{x:34,y:302,z:-82}, mid:{x:-33,y:276,z:-81}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S10" },
  { type:"avenida", name:"Av. Agro 1", from:{x:150,y:-180,z:-120}, to:{x:294,y:-147,z:-141}, mid:{x:222,y:-164,z:-131}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S11" },
  { type:"avenida", name:"Av. Finanças 1", from:{x:-250,y:-100,z:-80}, to:{x:-109,y:-51,z:-50}, mid:{x:-180,y:-76,z:-65}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S12" },
  { type:"avenida", name:"Av. Habitação 1", from:{x:100,y:200,z:120}, to:{x:243,y:254,z:119}, mid:{x:171,y:227,z:119}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S13" },
  { type:"avenida", name:"Av. Habitação 2", from:{x:100,y:200,z:120}, to:{x:-58,y:107,z:93}, mid:{x:21,y:153,z:106}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S13" },
  { type:"avenida", name:"Av. Saneamento 1", from:{x:-80,y:80,z:150}, to:{x:29,y:104,z:157}, mid:{x:-26,y:92,z:153}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S14" },
  { type:"avenida", name:"Av. Saneamento 2", from:{x:-80,y:80,z:150}, to:{x:-218,y:3,z:129}, mid:{x:-149,y:41,z:139}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S14" },
  { type:"avenida", name:"Av. Segurança 1", from:{x:50,y:-50,z:-200}, to:{x:223,y:15,z:-208}, mid:{x:136,y:-18,z:-204}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S15" },
  { type:"avenida", name:"Av. Segurança 2", from:{x:50,y:-50,z:-200}, to:{x:-138,y:-126,z:-168}, mid:{x:-44,y:-88,z:-184}, width:1.5, color:"#60a5fa", speed:60, capacity:"média", sector:"S15" },
  { type:"rua", name:"Rua E0101-E0102", from:{x:46,y:-201,z:-11}, to:{x:-41,y:-254,z:1}, mid:{x:-6,y:-222,z:-3}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E01" },
  { type:"rua", name:"Rua E0101-E0103", from:{x:46,y:-201,z:-11}, to:{x:45,y:-150,z:-13}, mid:{x:45,y:-180,z:-12}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E01" },
  { type:"rua", name:"Rua E0102-E0103", from:{x:-41,y:-254,z:1}, to:{x:45,y:-150,z:-13}, mid:{x:3,y:-197,z:2}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E01" },
  { type:"rua", name:"Rua E0201-E0202", from:{x:-327,y:-12,z:-14}, to:{x:-247,y:61,z:-100}, mid:{x:-285,y:30,z:-53}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E02" },
  { type:"rua", name:"Rua E0201-E0203", from:{x:-327,y:-12,z:-14}, to:{x:-311,y:57,z:-9}, mid:{x:-321,y:24,z:-11}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E02" },
  { type:"rua", name:"Rua E0202-E0203", from:{x:-247,y:61,z:-100}, to:{x:-311,y:57,z:-9}, mid:{x:-287,y:57,z:-60}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E02" },
  { type:"rua", name:"Rua E0301-E0302", from:{x:301,y:-53,z:59}, to:{x:256,y:-143,z:15}, mid:{x:272,y:-104,z:40}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E03" },
  { type:"rua", name:"Rua E0301-E0303", from:{x:301,y:-53,z:59}, to:{x:198,y:-37,z:54}, mid:{x:240,y:-49,z:49}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E03" },
  { type:"rua", name:"Rua E0302-E0303", from:{x:256,y:-143,z:15}, to:{x:198,y:-37,z:54}, mid:{x:233,y:-100,z:29}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E03" },
  { type:"rua", name:"Rua E0302-E0304", from:{x:256,y:-143,z:15}, to:{x:192,y:-23,z:46}, mid:{x:234,y:-89,z:30}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E03" },
  { type:"rua", name:"Rua E0303-E0304", from:{x:198,y:-37,z:54}, to:{x:192,y:-23,z:46}, mid:{x:190,y:-20,z:52}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E03" },
  { type:"rua", name:"Rua E0401-E0402", from:{x:231,y:173,z:-119}, to:{x:138,y:174,z:-103}, mid:{x:175,y:179,z:-107}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E04" },
  { type:"rua", name:"Rua E0401-E0403", from:{x:231,y:173,z:-119}, to:{x:163,y:218,z:-98}, mid:{x:189,y:194,z:-115}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E04" },
  { type:"rua", name:"Rua E0402-E0403", from:{x:138,y:174,z:-103}, to:{x:163,y:218,z:-98}, mid:{x:151,y:200,z:-109}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E04" },
  { type:"rua", name:"Rua E0501-E0502", from:{x:-140,y:135,z:119}, to:{x:-244,y:178,z:33}, mid:{x:-201,y:160,z:72}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E05" },
  { type:"rua", name:"Rua E0501-E0503", from:{x:-140,y:135,z:119}, to:{x:-230,y:179,z:86}, mid:{x:-180,y:159,z:96}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E05" },
  { type:"rua", name:"Rua E0502-E0503", from:{x:-244,y:178,z:33}, to:{x:-230,y:179,z:86}, mid:{x:-234,y:168,z:54}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E05" },
  { type:"rua", name:"Rua E0601-E0602", from:{x:104,y:-237,z:-95}, to:{x:113,y:-277,z:-31}, mid:{x:99,y:-255,z:-67}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E06" },
  { type:"rua", name:"Rua E0601-E0603", from:{x:104,y:-237,z:-95}, to:{x:67,y:-260,z:-97}, mid:{x:76,y:-239,z:-96}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E06" },
  { type:"rua", name:"Rua E0602-E0603", from:{x:113,y:-277,z:-31}, to:{x:67,y:-260,z:-97}, mid:{x:99,y:-271,z:-67}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E06" },
  { type:"rua", name:"Rua E0701-E0702", from:{x:-60,y:145,z:-127}, to:{x:64,y:53,z:-118}, mid:{x:9,y:91,z:-129}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E07" },
  { type:"rua", name:"Rua E0701-E0703", from:{x:-60,y:145,z:-127}, to:{x:-10,y:143,z:-122}, mid:{x:-44,y:151,z:-121}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E07" },
  { type:"rua", name:"Rua E0702-E0703", from:{x:64,y:53,z:-118}, to:{x:-10,y:143,z:-122}, mid:{x:33,y:90,z:-123}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E07" },
  { type:"rua", name:"Rua E0801-E0802", from:{x:-151,y:-105,z:115}, to:{x:-185,y:-100,z:107}, mid:{x:-163,y:-107,z:113}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E08" },
  { type:"rua", name:"Rua E0801-E0803", from:{x:-151,y:-105,z:115}, to:{x:-159,y:-184,z:87}, mid:{x:-155,y:-140,z:100}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E08" },
  { type:"rua", name:"Rua E0802-E0803", from:{x:-185,y:-100,z:107}, to:{x:-159,y:-184,z:87}, mid:{x:-176,y:-137,z:102}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E08" },
  { type:"rua", name:"Rua E0901-E0902", from:{x:256,y:115,z:-33}, to:{x:314,y:99,z:-35}, mid:{x:293,y:109,z:-36}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E09" },
  { type:"rua", name:"Rua E0901-E0903", from:{x:256,y:115,z:-33}, to:{x:319,y:144,z:-45}, mid:{x:289,y:122,z:-41}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E09" },
  { type:"rua", name:"Rua E0902-E0903", from:{x:314,y:99,z:-35}, to:{x:319,y:144,z:-45}, mid:{x:325,y:131,z:-39}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E09" },
  { type:"rua", name:"Rua E1001-E1002", from:{x:-162,y:219,z:-44}, to:{x:-55,y:224,z:-39}, mid:{x:-99,y:221,z:-40}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E10" },
  { type:"rua", name:"Rua E1001-E1003", from:{x:-162,y:219,z:-44}, to:{x:-154,y:259,z:-80}, mid:{x:-148,y:242,z:-64}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E10" },
  { type:"rua", name:"Rua E1002-E1003", from:{x:-55,y:224,z:-39}, to:{x:-154,y:259,z:-80}, mid:{x:-99,y:247,z:-57}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E10" },
  { type:"rua", name:"Rua E1101-E1102", from:{x:152,y:-230,z:-138}, to:{x:219,y:-176,z:-88}, mid:{x:195,y:-207,z:-113}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E11" },
  { type:"rua", name:"Rua E1101-E1103", from:{x:152,y:-230,z:-138}, to:{x:92,y:-176,z:-121}, mid:{x:123,y:-206,z:-133}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E11" },
  { type:"rua", name:"Rua E1102-E1103", from:{x:219,y:-176,z:-88}, to:{x:92,y:-176,z:-121}, mid:{x:156,y:-182,z:-105}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E11" },
  { type:"rua", name:"Rua E1201-E1202", from:{x:-247,y:-152,z:-81}, to:{x:-239,y:-129,z:-80}, mid:{x:-237,y:-137,z:-80}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E12" },
  { type:"rua", name:"Rua E1201-E1203", from:{x:-247,y:-152,z:-81}, to:{x:-186,y:-73,z:-112}, mid:{x:-218,y:-108,z:-98}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E12" },
  { type:"rua", name:"Rua E1202-E1203", from:{x:-239,y:-129,z:-80}, to:{x:-186,y:-73,z:-112}, mid:{x:-220,y:-96,z:-102}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E12" },
  { type:"rua", name:"Rua E1301-E1302", from:{x:93,y:201,z:141}, to:{x:81,y:267,z:104}, mid:{x:85,y:233,z:119}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E13" },
  { type:"rua", name:"Rua E1301-E1303", from:{x:93,y:201,z:141}, to:{x:152,y:181,z:166}, mid:{x:112,y:190,z:158}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E13" },
  { type:"rua", name:"Rua E1302-E1303", from:{x:81,y:267,z:104}, to:{x:152,y:181,z:166}, mid:{x:119,y:230,z:143}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E13" },
  { type:"rua", name:"Rua E1401-E1402", from:{x:-65,y:82,z:158}, to:{x:-126,y:57,z:117}, mid:{x:-87,y:72,z:141}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E14" },
  { type:"rua", name:"Rua E1401-E1403", from:{x:-65,y:82,z:158}, to:{x:-111,y:126,z:195}, mid:{x:-95,y:108,z:179}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E14" },
  { type:"rua", name:"Rua E1402-E1403", from:{x:-126,y:57,z:117}, to:{x:-111,y:126,z:195}, mid:{x:-114,y:96,z:160}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E14" },
  { type:"rua", name:"Rua E1501-E1502", from:{x:93,y:-91,z:-163}, to:{x:26,y:-86,z:-152}, mid:{x:54,y:-87,z:-154}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E15" },
  { type:"rua", name:"Rua E1501-E1503", from:{x:93,y:-91,z:-163}, to:{x:50,y:-72,z:-158}, mid:{x:78,y:-79,z:-161}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E15" },
  { type:"rua", name:"Rua E1502-E1503", from:{x:26,y:-86,z:-152}, to:{x:50,y:-72,z:-158}, mid:{x:28,y:-70,z:-154}, width:0.8, color:"#475569", speed:30, capacity:"baixa", sector:"E15" },
  { type:"ciclovia", name:"Ciclovia E01", from:{x:46,y:-201,z:-11}, to:{x:-41,y:-254,z:1}, mid:{x:2,y:-228,z:-5}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E01" },
  { type:"ciclovia", name:"Ciclovia E01", from:{x:-41,y:-254,z:1}, to:{x:45,y:-150,z:-13}, mid:{x:2,y:-202,z:-6}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E01" },
  { type:"ciclovia", name:"Ciclovia E01", from:{x:45,y:-150,z:-13}, to:{x:46,y:-201,z:-11}, mid:{x:45,y:-176,z:-12}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E01" },
  { type:"ciclovia", name:"Ciclovia E02", from:{x:-327,y:-12,z:-14}, to:{x:-247,y:61,z:-100}, mid:{x:-287,y:24,z:-57}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E02" },
  { type:"ciclovia", name:"Ciclovia E02", from:{x:-247,y:61,z:-100}, to:{x:-311,y:57,z:-9}, mid:{x:-279,y:59,z:-55}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E02" },
  { type:"ciclovia", name:"Ciclovia E02", from:{x:-311,y:57,z:-9}, to:{x:-327,y:-12,z:-14}, mid:{x:-319,y:22,z:-12}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E02" },
  { type:"ciclovia", name:"Ciclovia E03", from:{x:301,y:-53,z:59}, to:{x:256,y:-143,z:15}, mid:{x:278,y:-98,z:37}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E03" },
  { type:"ciclovia", name:"Ciclovia E03", from:{x:256,y:-143,z:15}, to:{x:198,y:-37,z:54}, mid:{x:227,y:-90,z:34}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E03" },
  { type:"ciclovia", name:"Ciclovia E03", from:{x:198,y:-37,z:54}, to:{x:192,y:-23,z:46}, mid:{x:195,y:-30,z:50}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E03" },
  { type:"ciclovia", name:"Ciclovia E03", from:{x:192,y:-23,z:46}, to:{x:301,y:-53,z:59}, mid:{x:246,y:-38,z:52}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E03" },
  { type:"ciclovia", name:"Ciclovia E04", from:{x:231,y:173,z:-119}, to:{x:138,y:174,z:-103}, mid:{x:184,y:173,z:-111}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E04" },
  { type:"ciclovia", name:"Ciclovia E04", from:{x:138,y:174,z:-103}, to:{x:163,y:218,z:-98}, mid:{x:150,y:196,z:-101}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E04" },
  { type:"ciclovia", name:"Ciclovia E04", from:{x:163,y:218,z:-98}, to:{x:231,y:173,z:-119}, mid:{x:197,y:195,z:-109}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E04" },
  { type:"ciclovia", name:"Ciclovia E05", from:{x:-140,y:135,z:119}, to:{x:-244,y:178,z:33}, mid:{x:-192,y:156,z:76}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E05" },
  { type:"ciclovia", name:"Ciclovia E05", from:{x:-244,y:178,z:33}, to:{x:-230,y:179,z:86}, mid:{x:-237,y:178,z:59}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E05" },
  { type:"ciclovia", name:"Ciclovia E05", from:{x:-230,y:179,z:86}, to:{x:-140,y:135,z:119}, mid:{x:-185,y:157,z:102}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E05" },
  { type:"ciclovia", name:"Ciclovia E06", from:{x:104,y:-237,z:-95}, to:{x:113,y:-277,z:-31}, mid:{x:108,y:-257,z:-63}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E06" },
  { type:"ciclovia", name:"Ciclovia E06", from:{x:113,y:-277,z:-31}, to:{x:67,y:-260,z:-97}, mid:{x:90,y:-269,z:-64}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E06" },
  { type:"ciclovia", name:"Ciclovia E06", from:{x:67,y:-260,z:-97}, to:{x:104,y:-237,z:-95}, mid:{x:85,y:-249,z:-96}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E06" },
  { type:"ciclovia", name:"Ciclovia E07", from:{x:-60,y:145,z:-127}, to:{x:64,y:53,z:-118}, mid:{x:2,y:99,z:-123}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E07" },
  { type:"ciclovia", name:"Ciclovia E07", from:{x:64,y:53,z:-118}, to:{x:-10,y:143,z:-122}, mid:{x:27,y:98,z:-120}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E07" },
  { type:"ciclovia", name:"Ciclovia E07", from:{x:-10,y:143,z:-122}, to:{x:-60,y:145,z:-127}, mid:{x:-35,y:144,z:-125}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E07" },
  { type:"ciclovia", name:"Ciclovia E08", from:{x:-151,y:-105,z:115}, to:{x:-185,y:-100,z:107}, mid:{x:-168,y:-103,z:111}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E08" },
  { type:"ciclovia", name:"Ciclovia E08", from:{x:-185,y:-100,z:107}, to:{x:-159,y:-184,z:87}, mid:{x:-172,y:-142,z:97}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E08" },
  { type:"ciclovia", name:"Ciclovia E08", from:{x:-159,y:-184,z:87}, to:{x:-151,y:-105,z:115}, mid:{x:-155,y:-145,z:101}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E08" },
  { type:"ciclovia", name:"Ciclovia E09", from:{x:256,y:115,z:-33}, to:{x:314,y:99,z:-35}, mid:{x:285,y:107,z:-34}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E09" },
  { type:"ciclovia", name:"Ciclovia E09", from:{x:314,y:99,z:-35}, to:{x:319,y:144,z:-45}, mid:{x:316,y:121,z:-40}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E09" },
  { type:"ciclovia", name:"Ciclovia E09", from:{x:319,y:144,z:-45}, to:{x:256,y:115,z:-33}, mid:{x:287,y:129,z:-39}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E09" },
  { type:"ciclovia", name:"Ciclovia E10", from:{x:-162,y:219,z:-44}, to:{x:-55,y:224,z:-39}, mid:{x:-109,y:221,z:-42}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E10" },
  { type:"ciclovia", name:"Ciclovia E10", from:{x:-55,y:224,z:-39}, to:{x:-154,y:259,z:-80}, mid:{x:-105,y:241,z:-60}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E10" },
  { type:"ciclovia", name:"Ciclovia E10", from:{x:-154,y:259,z:-80}, to:{x:-162,y:219,z:-44}, mid:{x:-158,y:239,z:-62}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E10" },
  { type:"ciclovia", name:"Ciclovia E11", from:{x:152,y:-230,z:-138}, to:{x:219,y:-176,z:-88}, mid:{x:185,y:-203,z:-113}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E11" },
  { type:"ciclovia", name:"Ciclovia E11", from:{x:219,y:-176,z:-88}, to:{x:92,y:-176,z:-121}, mid:{x:155,y:-176,z:-105}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E11" },
  { type:"ciclovia", name:"Ciclovia E11", from:{x:92,y:-176,z:-121}, to:{x:152,y:-230,z:-138}, mid:{x:122,y:-203,z:-130}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E11" },
  { type:"ciclovia", name:"Ciclovia E12", from:{x:-247,y:-152,z:-81}, to:{x:-239,y:-129,z:-80}, mid:{x:-243,y:-141,z:-81}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E12" },
  { type:"ciclovia", name:"Ciclovia E12", from:{x:-239,y:-129,z:-80}, to:{x:-186,y:-73,z:-112}, mid:{x:-213,y:-101,z:-96}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E12" },
  { type:"ciclovia", name:"Ciclovia E12", from:{x:-186,y:-73,z:-112}, to:{x:-247,y:-152,z:-81}, mid:{x:-217,y:-113,z:-97}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E12" },
  { type:"ciclovia", name:"Ciclovia E13", from:{x:93,y:201,z:141}, to:{x:81,y:267,z:104}, mid:{x:87,y:234,z:122}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E13" },
  { type:"ciclovia", name:"Ciclovia E13", from:{x:81,y:267,z:104}, to:{x:152,y:181,z:166}, mid:{x:116,y:224,z:135}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E13" },
  { type:"ciclovia", name:"Ciclovia E13", from:{x:152,y:181,z:166}, to:{x:93,y:201,z:141}, mid:{x:122,y:191,z:153}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E13" },
  { type:"ciclovia", name:"Ciclovia E14", from:{x:-65,y:82,z:158}, to:{x:-126,y:57,z:117}, mid:{x:-96,y:69,z:137}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E14" },
  { type:"ciclovia", name:"Ciclovia E14", from:{x:-126,y:57,z:117}, to:{x:-111,y:126,z:195}, mid:{x:-119,y:91,z:156}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E14" },
  { type:"ciclovia", name:"Ciclovia E14", from:{x:-111,y:126,z:195}, to:{x:-65,y:82,z:158}, mid:{x:-88,y:104,z:176}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E14" },
  { type:"ciclovia", name:"Ciclovia E15", from:{x:93,y:-91,z:-163}, to:{x:26,y:-86,z:-152}, mid:{x:59,y:-89,z:-158}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E15" },
  { type:"ciclovia", name:"Ciclovia E15", from:{x:26,y:-86,z:-152}, to:{x:50,y:-72,z:-158}, mid:{x:38,y:-79,z:-155}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E15" },
  { type:"ciclovia", name:"Ciclovia E15", from:{x:50,y:-72,z:-158}, to:{x:93,y:-91,z:-163}, mid:{x:71,y:-82,z:-161}, width:0.4, color:"#10b981", speed:15, capacity:"baixa", sector:"E15" }
];

const CITY_AIR_ROUTES = [
  { name:"Linha Saúde-Tech", from:"S01", to:"S05", from_pos:{x:0,y:-380,z:0}, to_pos:{x:-180,y:0,z:80}, mid:{x:-90,y:-420,z:40}, alt:180, color:"#f472b6", aircraft:"heli", freq:19 },
  { name:"Linha Comércio-Finanças", from:"S03", to:"S12", from_pos:{x:250,y:-240,z:60}, to_pos:{x:-250,y:-260,z:-80}, mid:{x:0,y:-300,z:-10}, alt:160, color:"#f472b6", aircraft:"aviação", freq:19 },
  { name:"Linha Governo-Saúde", from:"S07", to:"S01", from_pos:{x:0,y:-100,z:-150}, to_pos:{x:0,y:-400,z:0}, mid:{x:0,y:-440,z:-75}, alt:200, color:"#f472b6", aircraft:"aviação", freq:23 },
  { name:"Linha Indústria-Energia", from:"S04", to:"S09", from_pos:{x:200,y:10,z:-80}, to_pos:{x:300,y:-40,z:0}, mid:{x:250,y:-80,z:-40}, alt:140, color:"#f472b6", aircraft:"drone", freq:23 },
  { name:"Linha Tech-Cultura", from:"S05", to:"S10", from_pos:{x:-180,y:50,z:80}, to_pos:{x:-100,y:120,z:-80}, mid:{x:-140,y:10,z:0}, alt:130, color:"#f472b6", aircraft:"drone", freq:17 },
  { name:"Linha Transporte-Comércio", from:"S08", to:"S03", from_pos:{x:-150,y:-300,z:100}, to_pos:{x:250,y:-230,z:60}, mid:{x:50,y:-340,z:80}, alt:150, color:"#f472b6", aircraft:"heli", freq:16 },
  { name:"Linha Agro-Saneamento", from:"S11", to:"S14", from_pos:{x:150,y:-350,z:-120}, to_pos:{x:-80,y:-90,z:150}, mid:{x:35,y:-390,z:15}, alt:170, color:"#f472b6", aircraft:"aviação", freq:13 },
  { name:"Linha Habitação-Governo", from:"S13", to:"S07", from_pos:{x:100,y:80,z:120}, to_pos:{x:0,y:-20,z:-150}, mid:{x:50,y:-60,z:-15}, alt:120, color:"#f472b6", aircraft:"aviação", freq:8 },
  { name:"Linha Serviços-Educação", from:"S06", to:"S02", from_pos:{x:100,y:-360,z:-60}, to_pos:{x:-280,y:-110,z:-50}, mid:{x:-90,y:-400,z:-55}, alt:110, color:"#f472b6", aircraft:"heli", freq:16 },
  { name:"Linha Segurança-Indústria", from:"S15", to:"S04", from_pos:{x:50,y:-195,z:-200}, to_pos:{x:200,y:5,z:-80}, mid:{x:125,y:-235,z:-140}, alt:145, color:"#f472b6", aircraft:"drone", freq:8 }
];

const CITY_TRANSIT_STOPS = [
  { eid:"E0101", x:46, y:-201, z:-11, type:"parada", lines:2 },
  { eid:"E0102", x:-41, y:-254, z:1, type:"terminal", lines:1 },
  { eid:"E0103", x:45, y:-150, z:-13, type:"parada", lines:3 },
  { eid:"E0201", x:-327, y:-12, z:-14, type:"parada", lines:3 },
  { eid:"E0202", x:-247, y:61, z:-100, type:"terminal", lines:3 },
  { eid:"E0203", x:-311, y:57, z:-9, type:"terminal", lines:2 },
  { eid:"E0301", x:301, y:-53, z:59, type:"parada", lines:4 },
  { eid:"E0302", x:256, y:-143, z:15, type:"ponto", lines:1 },
  { eid:"E0303", x:198, y:-37, z:54, type:"parada", lines:1 },
  { eid:"E0304", x:192, y:-23, z:46, type:"ponto", lines:4 },
  { eid:"E0401", x:231, y:173, z:-119, type:"terminal", lines:2 },
  { eid:"E0402", x:138, y:174, z:-103, type:"ponto", lines:1 },
  { eid:"E0403", x:163, y:218, z:-98, type:"parada", lines:1 },
  { eid:"E0501", x:-140, y:135, z:119, type:"ponto", lines:4 },
  { eid:"E0502", x:-244, y:178, z:33, type:"terminal", lines:1 },
  { eid:"E0503", x:-230, y:179, z:86, type:"terminal", lines:1 },
  { eid:"E0601", x:104, y:-237, z:-95, type:"ponto", lines:3 },
  { eid:"E0602", x:113, y:-277, z:-31, type:"parada", lines:4 },
  { eid:"E0603", x:67, y:-260, z:-97, type:"parada", lines:4 },
  { eid:"E0701", x:-60, y:145, z:-127, type:"ponto", lines:1 },
  { eid:"E0702", x:64, y:53, z:-118, type:"terminal", lines:3 },
  { eid:"E0703", x:-10, y:143, z:-122, type:"ponto", lines:2 },
  { eid:"E0801", x:-151, y:-105, z:115, type:"ponto", lines:1 },
  { eid:"E0802", x:-185, y:-100, z:107, type:"ponto", lines:2 },
  { eid:"E0803", x:-159, y:-184, z:87, type:"terminal", lines:2 },
  { eid:"E0901", x:256, y:115, z:-33, type:"parada", lines:3 },
  { eid:"E0902", x:314, y:99, z:-35, type:"parada", lines:1 },
  { eid:"E0903", x:319, y:144, z:-45, type:"ponto", lines:1 },
  { eid:"E1001", x:-162, y:219, z:-44, type:"ponto", lines:2 },
  { eid:"E1002", x:-55, y:224, z:-39, type:"parada", lines:4 },
  { eid:"E1003", x:-154, y:259, z:-80, type:"ponto", lines:4 },
  { eid:"E1101", x:152, y:-230, z:-138, type:"terminal", lines:1 },
  { eid:"E1102", x:219, y:-176, z:-88, type:"ponto", lines:4 },
  { eid:"E1103", x:92, y:-176, z:-121, type:"parada", lines:4 },
  { eid:"E1201", x:-247, y:-152, z:-81, type:"ponto", lines:4 },
  { eid:"E1202", x:-239, y:-129, z:-80, type:"ponto", lines:4 },
  { eid:"E1203", x:-186, y:-73, z:-112, type:"terminal", lines:3 },
  { eid:"E1301", x:93, y:201, z:141, type:"terminal", lines:1 },
  { eid:"E1302", x:81, y:267, z:104, type:"ponto", lines:4 },
  { eid:"E1303", x:152, y:181, z:166, type:"terminal", lines:3 },
  { eid:"E1401", x:-65, y:82, z:158, type:"ponto", lines:3 },
  { eid:"E1402", x:-126, y:57, z:117, type:"ponto", lines:3 },
  { eid:"E1403", x:-111, y:126, z:195, type:"parada", lines:4 },
  { eid:"E1501", x:93, y:-91, z:-163, type:"parada", lines:3 },
  { eid:"E1502", x:26, y:-86, z:-152, type:"terminal", lines:1 },
  { eid:"E1503", x:50, y:-72, z:-158, type:"terminal", lines:1 }
];
