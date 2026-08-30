#!/usr/bin/env bash
################################################################################
# Fig 4 | snpEff annotation of somatic calls
################################################################################

snpeff_jar="tools/snpEff/snpEff.jar"
annovar_path="results/annovar"

genome="GRCh37.75"

for cell_dir in "${annovar_path}"/*; do
  [ -d "$cell_dir" ] || continue

  for muttype in snv indel; do

    input_vcf="${cell_dir}/pass_${muttype}.hg19_multianno.vcf"
    output_vcf="${cell_dir}/pass_${muttype}.hg19_snpeff.vcf"

    [ -f "$input_vcf" ] || continue
    [ -s "$output_vcf" ] && continue

    java -Xmx16g -jar "$snpeff_jar" -v "$genome" "$input_vcf" > "$output_vcf"

  done
done
