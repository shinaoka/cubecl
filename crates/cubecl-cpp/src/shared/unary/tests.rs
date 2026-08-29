use crate::{
    cuda::{CudaDialect, mma::PtxWmmaCompiler},
    shared::{Elem, Instruction, Item, UnaryInstruction, Variable},
};
use cubecl_core::ir::{ConstantValue, ElemType};

type Dialect = CudaDialect<PtxWmmaCompiler>;

fn cast_variable(input: Variable<Dialect>, output: Elem<Dialect>) -> String {
    Instruction::Assign(UnaryInstruction {
        input,
        out: Variable::Named {
            name: "output",
            item: Item::scalar(output, false),
        },
    })
    .to_string()
}

fn cast(input: Elem<Dialect>, output: Elem<Dialect>) -> String {
    cast_variable(
        Variable::Named {
            name: "input",
            item: Item::scalar(input, false),
        },
        output,
    )
}

#[test]
fn cuda_complex_casts_use_cucomplex_components_and_constructors() {
    assert_eq!(
        cast_variable(
            Variable::Constant(ConstantValue::UInt(0), Item::scalar(Elem::U32, false)),
            Elem::CF64,
        ),
        "output = make_cuDoubleComplex(uint32(0), 0.0);\n"
    );
    assert_eq!(
        cast(Elem::F64, Elem::CF32),
        "output = make_cuFloatComplex(input, 0.0f);\n"
    );
    assert_eq!(
        cast(Elem::CF64, Elem::CF32),
        "output = make_cuFloatComplex(input.x, input.y);\n"
    );
    assert_eq!(
        cast(Elem::CF32, Elem::CF64),
        "output = make_cuDoubleComplex(input.x, input.y);\n"
    );
    assert_eq!(cast(Elem::CF32, Elem::F64), "output = double(input.x);\n");
    assert_eq!(
        cast(Elem::CF32, Elem::Bool),
        "output = (input.x != 0 || input.y != 0);\n"
    );
    assert_eq!(
        ConstantValue::Complex(0.0, 1.0).cast_to(ElemType::Bool),
        ConstantValue::Bool(true)
    );
    assert_eq!(
        ConstantValue::Complex(0.0, 0.0).cast_to(ElemType::Bool),
        ConstantValue::Bool(false)
    );
}
