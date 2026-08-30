#!/usr/bin/env bash
################################################################################
# Fig 4 | ANNOVAR annotation of somatic calls
################################################################################

input_file="data/GN_PTA_snv_indel_pass.FILTERED.txt"
output_path="results/annovar"

annovar_dir="tools/annovar"
table_annovar="${annovar_dir}/table_annovar.pl"
humandb_dir="${annovar_dir}/humandb"
xref="${annovar_dir}/example/gene_xref.txt"

build="hg19"
threads=20

cells=$(awk -F',' 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "cell") s = i; next } { print $s }' \
          "$input_file" | sort -u)

for cell in $cells; do
  for muttype in snv indel; do

    out_dir="${output_path}/${cell}"
    prefix="${out_dir}/pass_${muttype}"
    vcf="${prefix}.vcf"
    mkdir -p "$out_dir"

    # PASS calls of this cell and mutation type, as a minimal VCF
    awk -F',' -v s="$cell" -v mt="$muttype" '
      BEGIN {
        OFS = "\t"
        print "##fileformat=VCFv4.2"
        print "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"
      }
      NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
      $col["pass"] == "TRUE" && $col["cell"] == s && $col["muttype"] == mt {
        print $col["chr"], $col["pos"], s, $col["refnt"], $col["altnt"], mt, "PASS", $col["mutsig"]
        n++
      }
      END { exit (n > 0 ? 0 : 1) }
    ' "$input_file" > "$vcf" || { rm -f "$vcf"; continue; }

    perl "$table_annovar" "$vcf" "$humandb_dir" \
      -buildver "$build" \
      -out "$prefix" \
      -remove \
      -protocol refGene \
      -operation g \
      -nastring . \
      -vcfinput \
      -polish \
      -thread "$threads" \
      --maxgenethread "$threads"

    perl "$table_annovar" "${prefix}.avinput" "$humandb_dir" \
      -buildver "$build" \
      -out "$prefix" \
      -remove \
      -protocol refGene \
      -operation gx \
      -nastring . \
      -csvout \
      -polish \
      -xref "$xref" \
      -thread "$threads" \
      --maxgenethread "$threads"

  done
done
