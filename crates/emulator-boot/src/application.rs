//! Application-profile startup plans with caller-selected firmware and media.
use crate::{graph::ProcessorStart, xpp};
use fujimod_catalog_configuration::{RunEngine, RunInputs};
use fujimod_catalog_resolution::ResolvedConfiguration;
use fujimod_emulator::board::bus::Mode;
use fujimod_emulator_cpu::Register;
use fujimod_emulator_devices::graphics::scanout::Frame;
use fujimod_emulator_session::Session;
use fujimod_emulator_xp4_board as xp4;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// A checked emulation plan and explicit update file.
pub struct Startup {
    pub update: PathBuf,
    pub plan: PathBuf,
    pub discovery: bool,
    pub quantum: Option<usize>,
    pub card_image: Option<PathBuf>,
    pub card_files: Vec<(String, PathBuf)>,
}

/// Application startup facts. File references are relative to the plan's directory.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    configuration_id: String,
    flash: Option<PathBuf>,
    #[serde(default)]
    images: Vec<Image>,
    #[serde(default)]
    card_files: BTreeMap<String, PathBuf>,
    board: Plan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Image {
    address: u64,
    file: PathBuf,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Plan {
    /// Each core's reset registers and the entry labels the firmware image
    /// adds; the board itself comes from the catalog.
    Xp4 {
        registers: Vec<Vec<(Register, u64)>>,
        #[serde(default)]
        labels: Vec<u64>,
    },
    Xpp {
        registers: Vec<Vec<(Register, u64)>>,
        #[serde(default)]
        sub_cpu_registers: Vec<(u8, u8)>,
        #[serde(default)]
        assets: BTreeMap<String, PathBuf>,
    },
}

/// An application-profile board: a shared session for every processor.
pub enum Board {
    Shared(Box<Session>),
}

impl crate::host::Memory for Board {
    fn read(&self, address: u64, bytes: &mut [u8]) -> Result<(), String> {
        match self {
            Self::Shared(session) => session.board.inspect_physical(address, bytes),
        }
    }

    fn write(&self, address: u64, bytes: &[u8]) -> Result<(), String> {
        match self {
            Self::Shared(session) => session.board.write_physical(address, bytes),
        }
    }
}

impl crate::mod_loader::Machine for Board {
    fn arm32_core_count(&self) -> usize {
        match self {
            Self::Shared(session) => crate::mod_loader::BoardMachine(&session.board).arm32_core_count(),
        }
    }

    fn arm32_translate(
        &self,
        core: usize,
        address: u64,
        access: fujimod_emulator_cpu::Access,
    ) -> Result<u64, String> {
        match self {
            Self::Shared(session) => crate::mod_loader::BoardMachine(&session.board).arm32_translate(core, address, access),
        }
    }

    fn arm32_mmu_enabled(&self, core: usize) -> Result<bool, String> {
        match self {
            Self::Shared(session) => crate::mod_loader::BoardMachine(&session.board).arm32_mmu_enabled(core),
        }
    }
}

impl Board {
    pub fn gpio_input(&mut self, port: u8, pin: u8, high: bool) -> Result<(), String> {
        match self {
            Self::Shared(session) => fujimod_emulator_xpp::board::gpio_input(session, port, pin, high),
        }
    }

    pub fn set_scene(&mut self, scene: &Value) -> Result<(), String> {
        match self {
            Self::Shared(session) => session.board.set_scene(scene),
        }
    }
}

pub struct Started {
    pub board: Board,
    pub quantum: usize,
    pub ticks_per_second: u64,
    pub engine: RunEngine,
}

impl crate::host::Execution for Started {
    fn progress(&self) -> (Vec<u64>, u64) {
        Started::progress(self)
    }

    fn run_window(
        &mut self,
        end: u64,
        quantum: usize,
        pause_after: Option<u64>,
        until: Option<u64>,
    ) -> Result<(), String> {
        match &mut self.board {
            Board::Shared(session) => session.run_window(end, quantum, pause_after, until),
        }
    }
}

impl Started {
    pub fn progress(&self) -> (Vec<u64>, u64) {
        match &self.board {
            Board::Shared(session) => (session.board.instruction_counts(), session.board.now()),
        }
    }

    pub fn load_snapshot(&mut self, path: &Path) -> Result<(), String> {
        match &mut self.board {
            Board::Shared(session) => crate::load_snapshot(session, path),
        }
    }

    pub fn save_snapshot(&self, path: &Path) -> Result<(), String> {
        match &self.board {
            Board::Shared(session) => crate::save_snapshot(session, path),
        }
    }

    pub fn display_frames(&self) -> Vec<Frame> {
        match &self.board {
            Board::Shared(session) => session.board.display_frames(),
        }
    }

    /// Drive each scheduler window to the next requested host observation or stop.
    pub fn run_with(
        &mut self,
        instruction_limit: Option<u64>,
        duration: Option<std::time::Duration>,
        mut observe: impl FnMut(&mut Self) -> Result<Option<u64>, String>,
    ) -> Result<Value, String> {
        if instruction_limit.is_none() && duration.is_none() {
            return Err("application startup requires an instruction or time limit".into());
        }
        let limits =
            crate::host::Limits::new(self, instruction_limit, duration, self.ticks_per_second)?;
        limits.drive(
            self,
            self.quantum,
            |board| observe(board).map(std::ops::ControlFlow::Continue),
            |_, _| Ok(()),
        )?;
        let (instructions, now) = self.progress();
        let state = match &self.board {
            Board::Shared(session) => session.board.state(),
        };
        let stubs = match &self.board {
            Board::Shared(session) => session.board.discovery_callee_stubs(),
        };
        Ok(
            json!({"engine":self.engine,"instructions":instructions,"now":now,"state":state,"discovery_callee_stubs":stubs}),
        )
    }
}

fn bytes(directory: &Path, path: &Path) -> Result<Vec<u8>, String> {
    let path = directory.join(path);
    let size = std::fs::metadata(&path)
        .map_err(|error| format!("{}: {error}", path.display()))?
        .len();
    if size > 512 << 20 {
        return Err(format!(
            "{} exceeds the startup asset size bound",
            path.display()
        ));
    }
    std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))
}

fn named(
    directory: &Path,
    inputs: BTreeMap<String, PathBuf>,
) -> Result<BTreeMap<String, Vec<u8>>, String> {
    inputs
        .into_iter()
        .map(|(name, path)| Ok((name, bytes(directory, &path)?)))
        .collect()
}

pub fn open(resolved: &ResolvedConfiguration, startup: Startup) -> Result<Started, String> {
    let update = bytes(Path::new(""), &startup.update)?;
    let identity = resolved
        .reference
        .firmware
        .update
        .as_ref()
        .ok_or("firmware has no update identity")?;
    if update.len() as u64 != identity.size
        || format!("{:x}", Sha256::digest(&update)) != identity.sha256
    {
        return Err("application update differs from the catalog identity".into());
    }
    let document: Document = serde_json::from_slice(&bytes(Path::new(""), &startup.plan)?)
        .map_err(|error| format!("{}: {error}", startup.plan.display()))?;
    if document.configuration_id != resolved.configuration.id {
        return Err("application startup plan selects another run configuration".into());
    }
    let directory = startup.plan.parent().unwrap_or_else(|| Path::new(""));
    let flash = document
        .flash
        .as_deref()
        .map(|path| bytes(directory, path))
        .transpose()?;
    let mut files = named(directory, document.card_files)?;
    for (name, bytes) in crate::media::read_files(&startup.card_files)? {
        if files.insert(name.clone(), bytes).is_some() {
            return Err(format!("SD card file {name} supplied twice"));
        }
    }
    let card_image = startup
        .card_image
        .as_deref()
        .map(crate::media::read_image)
        .transpose()?;
    let execution = resolved.configuration.execution.as_ref();
    let engine = execution.map_or(RunEngine::Unicorn, |execution| execution.engine);
    let quantum = startup
        .quantum
        .or_else(|| execution.map(|run| run.quantum))
        .unwrap_or(1000);
    if quantum == 0 {
        return Err("processor quantum must be positive".into());
    }
    let mode = if startup.discovery {
        Mode::Discovery
    } else {
        Mode::Strict
    };
    let body = &resolved.reference;
    let board = match (body.soc.id.as_str(), document.board) {
        ("xp4", Plan::Xp4 { registers, labels }) => {
            fujimod_emulator::board::factory::validate_engine(engine, [fujimod_emulator_cpu::Model::CortexA7])?;
            let files = files.into_iter().collect::<Vec<_>>();
            let (card_files, media) = match card_image {
                Some(medium) => (Vec::new(), vec![(0, fujimod_emulator_session::card::merge_files(medium, &files)?)]),
                None => (files, Vec::new()),
            };
            Board::Shared(Box::new(xp4_session(resolved, Xp4Inputs {
                registers, labels: &labels, flash, card_files, media, mode, engine, quantum,
            })?))
        }
        (
            "xpp",
            Plan::Xpp {
                registers,
                sub_cpu_registers,
                assets,
            },
        ) => {
            if registers.len() != body.soc.processors.len() {
                return Err(
                    "X-Processor Pro startup plan core count differs from the selected SoC".into(),
                );
            }
            let starts = body
                .soc
                .processors
                .iter()
                .zip(registers)
                .map(|(processor, registers)| ProcessorStart {
                    processor_id: processor.id.clone(),
                    released: true,
                    parked_until_sgi: false,
                    registers,
                })
                .collect::<Vec<_>>();
            let mut assets = named(directory, assets)?;
            let assembly = xpp::assemble(
                resolved,
                &xpp::Inputs {
                    starts: &starts,
                    sub_cpu_registers: &sub_cpu_registers,
                    assets: &assets,
                    sensor_image: None,
                },
            )?;
            assets.insert(
                assembly.flash_asset.into(),
                flash.ok_or("X-Processor Pro requires flash backing")?,
            );
            let mut devices = assembly.devices;
            let mut files = files.into_iter().collect::<Vec<_>>();
            if card_image.is_none() {
                install_card_files(&mut devices, &mut assets, std::mem::take(&mut files))?;
            }
            let definition = json!({
                "configuration_id": resolved.configuration.id,
                "firmware_id": resolved.configuration.firmware_id,
                "engine": engine,
                "quantum": quantum,
                "mode": mode.name(),
                "execution": crate::graph::value(
                    &resolved.reference, &assembly.execution,
                    fujimod_catalog_configuration::RunInstructionRate {
                        numerator: assembly.rate.0,
                        denominator: assembly.rate.1,
                    },
                ),
                "devices": &devices,
            });
            let spec = serde_json::from_value(devices).map_err(|error| error.to_string())?;
            let mut board = fujimod_emulator_xpp::board::build(
                assembly.execution,
                spec,
                &assets,
                mode,
                engine,
            )?
            .into_session(definition)?;
            if let Some(medium) = card_image {
                let slot = board
                    .contacts
                    .keys()
                    .next()
                    .cloned()
                    .ok_or("board has no modeled SD card slot")?;
                board.install_card_files(&slot, Some(medium), &files)?;
            }
            Board::Shared(Box::new(board))
        }
        _ => return Err("application startup plan does not match the selected processor".into()),
    };
    for image in document.images {
        let data = bytes(directory, &image.file)?;
        if format!("{:x}", Sha256::digest(&data)) != image.sha256 {
            return Err(format!(
                "{} differs from its declared SHA-256",
                image.file.display()
            ));
        }
        match &board {
            Board::Shared(session) => session.board.write_memory(0, image.address, &data)?,
        }
    }
    let ticks_per_second = match resolved.configuration.inputs.as_ref() {
        Some(RunInputs::Xp4(inputs)) => inputs.ticks_per_second,
        _ => 54_000_000,
    };
    Ok(Started { board, quantum, ticks_per_second, engine })
}

/// The caller-owned inputs of an XP4 board: firmware-side reset state,
/// entry labels, flash contents and SD card media.
struct Xp4Inputs<'a> {
    registers: Vec<Vec<(Register, u64)>>,
    labels: &'a [u64],
    flash: Option<Vec<u8>>,
    card_files: Vec<(String, Vec<u8>)>,
    media: Vec<(usize, fujimod_emulator_devices::parts::sd_card::Card)>,
    mode: Mode,
    engine: RunEngine,
    quantum: usize,
}

/// An XP4 run on the shared board. The board comes from the catalog; the
/// declared instruction rate is each core's.
fn xp4_session(resolved: &ResolvedConfiguration, inputs: Xp4Inputs<'_>) -> Result<Session, String> {
    let processors = &resolved.reference.soc.processors;
    if inputs.registers.len() != processors.len() {
        return Err("XP4 startup plan core count differs from the selected SoC".into());
    }
    let starts: Vec<ProcessorStart> = processors.iter().zip(&inputs.registers)
        .map(|(processor, registers)| ProcessorStart {
            processor_id: processor.id.clone(), released: true, parked_until_sgi: false,
            registers: registers.clone(),
        }).collect();
    let definition = xp4::catalog::definition(resolved, inputs.registers, inputs.labels)?;
    let rate: (u64, u64) = serde_json::from_value(definition["instructions_per_tick"].clone())
        .map_err(|error| format!("XP4 instruction rate: {error}"))?;
    let rate = fujimod_catalog_configuration::RunInstructionRate {
        numerator: rate.0,
        denominator: rate.1,
    };
    let execution = crate::graph::assemble_with_starts(
        &resolved.reference, resolved.dram_address_space_limit_bytes()?, rate, None, &starts,
    )?;
    let record = json!({
        "configuration_id": resolved.configuration.id,
        "firmware_id": resolved.configuration.firmware_id,
        "engine": inputs.engine,
        "quantum": inputs.quantum,
        "mode": inputs.mode.name(),
        "execution": crate::graph::value(&resolved.reference, &execution, rate),
        "devices": &definition,
    });
    let definition = serde_json::from_value(definition).map_err(|error| format!("XP4 board definition: {error}"))?;
    xp4::shared::build(execution, definition, inputs.flash, inputs.card_files, inputs.media,
        inputs.mode, inputs.engine, record)
}

fn install_card_files(
    devices: &mut Value,
    assets: &mut BTreeMap<String, Vec<u8>>,
    files: Vec<(String, Vec<u8>)>,
) -> Result<(), String> {
    if files.is_empty() {
        return Ok(());
    }
    let host = devices["nodes"]
        .as_array_mut()
        .into_iter()
        .flatten()
        .find(|node| node["kind"] == "sd")
        .ok_or("X-Processor Pro startup has no modeled SD card host")?;
    let mut entries = BTreeMap::new();
    for (index, (name, bytes)) in files.into_iter().enumerate() {
        let key = format!("startup-sd-file-{index}");
        if assets.insert(key.clone(), bytes).is_some() {
            return Err("SD card file asset identity collides with another input".into());
        }
        entries.insert(name, key);
    }
    host["files"] = json!(entries);
    Ok(())
}

#[cfg(test)]
mod shared_board_tests {
    use super::*;

    /// Boots a prepared XP4 application plan on the shared board:
    /// `FUJIMOD_XP4_PLAN` names the plan and `FUJIMOD_XP4_INSTRUCTIONS` the
    /// instruction budget. Skipped without a plan; plans hold firmware bytes.
    #[test]
    fn xp4_plan_runs_on_the_shared_board() {
        let Ok(path) = std::env::var("FUJIMOD_XP4_PLAN") else { return };
        let count: u64 = std::env::var("FUJIMOD_XP4_INSTRUCTIONS").ok().and_then(|n| n.parse().ok()).unwrap_or(1_000_000);
        let plan = PathBuf::from(path);
        let document: Document = serde_json::from_slice(&std::fs::read(&plan).unwrap()).unwrap();
        let directory = plan.parent().unwrap();
        let resolved = ResolvedConfiguration::bundled(&document.configuration_id).unwrap();
        let Plan::Xp4 { registers, labels } = document.board else { panic!("not an XP4 plan") };
        let flash = document.flash.as_deref().map(|path| bytes(directory, path)).transpose().unwrap();
        let session = xp4_session(&resolved, Xp4Inputs {
            registers, labels: &labels, flash, card_files: Vec::new(), media: Vec::new(),
            mode: Mode::Strict, engine: RunEngine::Unicorn, quantum: 1000,
        }).unwrap();
        let mut board = session.board;
        let cores = board.instruction_counts().len();
        for image in &document.images {
            board.write_physical(image.address, &bytes(directory, &image.file).unwrap()).unwrap();
        }
        let result = board.run_for_instructions(count, 1000);
        let pcs: Vec<String> = (0..cores)
            .map(|core| format!("{:#x}", board.read_register(core, Register::Pc).unwrap_or(0))).collect();
        eprintln!("FUJIMOD shared XP4 {}: {:?} instructions {:?} now {} pcs {:?}",
            document.configuration_id, result, board.instruction_counts(), board.now(), pcs);
        if let Ok(range) = std::env::var("FUJIMOD_XP4_DUMP") {
            let (address, length) = range.split_once(':').unwrap();
            let address = u64::from_str_radix(address.trim_start_matches("0x"), 16).unwrap();
            let mut bytes = vec![0; length.parse().unwrap()];
            board.read_physical(address, &mut bytes).unwrap();
            let words: Vec<String> = bytes.chunks(4).map(|w| format!("{:08x}", u32::from_le_bytes(w.try_into().unwrap()))).collect();
            eprintln!("FUJIMOD shared XP4 dump {address:#x}: {}", words.join(" "));
        }
        if let Some(directory) = std::env::var_os("FUJIMOD_XP4_FRAMES") {
            for frame in board.display_frames() {
                let png = crate::png_writer::rgba_png(frame.width, frame.height, &frame.rgba).unwrap();
                std::fs::write(Path::new(&directory).join(format!("output{}.png", frame.output)), png).unwrap();
            }
        }
        if std::env::var_os("FUJIMOD_XP4_STATE").is_some() {
            eprintln!("FUJIMOD shared XP4 state {}", board.state());
        }
        result.unwrap();
    }
}
